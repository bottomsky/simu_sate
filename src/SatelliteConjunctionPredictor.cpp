#include "SatelliteConjunctionPredictor.h"

#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <tuple>
#include <algorithm>

#include "j2_orbit_propagator.h"

namespace {
struct CellKey {
    int x, y, z;
    bool operator==(const CellKey& other) const noexcept {
        return x==other.x && y==other.y && z==other.z;
    }
};

struct CellKeyHash {
    std::size_t operator()(const CellKey& k) const noexcept {
        std::size_t h1 = std::hash<int>{}(k.x);
        std::size_t h2 = std::hash<int>{}(k.y);
        std::size_t h3 = std::hash<int>{}(k.z);
        return h1 ^ (h2<<1) ^ (h3<<2);
    }
};

inline double sqr(double v) { return v*v; }
inline double norm2(const Eigen::Vector3d& v) { return v.squaredNorm(); }
inline double dist2(const Eigen::Vector3d& a, const Eigen::Vector3d& b) { return (a-b).squaredNorm(); }

double wrapPi(double angle) {
    angle = std::fmod(angle + M_PI, 2.0 * M_PI);
    if (angle < 0.0) angle += 2.0 * M_PI;
    return angle - M_PI;
}

struct Rates {
    double raandot;
    double argpdot;
    double meandot;
};

Rates computeRates(const CompactOrbitalElements& elem) {
    const double a = elem.a;
    const double e = elem.e;
    const double inc = elem.i;
    const double n = std::sqrt(MU / (a * a * a));
    const double p = std::max(1e-3, a * (1.0 - e * e));
    const double factor = 1.5 * J2 * n * std::pow(RE / p, 2.0);
    const double cosi = std::cos(inc);
    const double cos2 = cosi * cosi;
    const double sqrt_term = std::sqrt(std::max(0.0, 1.0 - e * e));

    Rates r{};
    r.raandot = -factor * cosi;
    r.argpdot = 0.5 * factor * (5.0 * cos2 - 1.0);
    r.meandot = n + 0.5 * factor * sqrt_term * (3.0 * cos2 - 1.0);
    return r;
}
}

std::vector<ConjunctionEvent> SatelliteConjunctionPredictor::predict(const J2ConstellationPropagator& propagator) const {
    // Choose strategy based on satellite count and horizon if auto
    J2ConstellationPropagator prop = propagator; // local copy so we can advance time
    const std::size_t n = prop.getSatelliteCount();

    ConjunctionStrategy strat = strategy_;
    if (strategy_ == ConjunctionStrategy::Auto) {
        if (n <= 200) strat = ConjunctionStrategy::Hierarchical;
        else strat = ConjunctionStrategy::SpatialGrid;
    }

    if (strat == ConjunctionStrategy::ConnectionTable) {
        ensureConnectionTable(propagator);
    }

    switch (strat) {
        case ConjunctionStrategy::Hierarchical:
            return predictHierarchical(std::move(prop));
        case ConjunctionStrategy::SpatialGrid:
            return predictSpatialGrid(std::move(prop));
        case ConjunctionStrategy::ConnectionTable:
            if (!connection_table_valid_) {
                // fallback if table could not be built
                return predictSpatialGrid(std::move(prop));
            }
            return predictConnectionTable(std::move(prop));
        case ConjunctionStrategy::Auto:
        default:
            return predictSpatialGrid(std::move(prop));
    }
}

std::vector<ConjunctionEvent> SatelliteConjunctionPredictor::predictSpatialGrid(J2ConstellationPropagator prop) const {
    std::vector<ConjunctionEvent> events;
    const std::size_t n = prop.getSatelliteCount();
    if (n < 2) return events;

    const double cell = cfg_.cell_size > 0 ? cfg_.cell_size : cfg_.threshold;
    const double thresh2 = cfg_.threshold * cfg_.threshold;

    std::unordered_set<std::uint64_t> seen_pairs; // i<<32 | j
    auto pair_id = [](std::size_t i, std::size_t j) -> std::uint64_t {
        if (i>j) std::swap(i,j);
        return (static_cast<std::uint64_t>(i)<<32) | static_cast<std::uint64_t>(j);
    };

    double cur = 0.0;
    while (cur < cfg_.horizon) {
        cur = std::min(cfg_.horizon, cur + cfg_.coarse_dt);
        prop.propagateConstellation(cur);
        // Collect positions
        auto pos = prop.getAllPositions(); // 3 x n
        std::unordered_map<CellKey, std::vector<std::size_t>, CellKeyHash> grid;
        grid.reserve(n*2);

        for (std::size_t i = 0; i < n; ++i) {
            Eigen::Vector3d p = pos.col(static_cast<Eigen::Index>(i));
            CellKey key{ static_cast<int>(std::floor(p.x()/cell)),
                        static_cast<int>(std::floor(p.y()/cell)),
                        static_cast<int>(std::floor(p.z()/cell)) };
            grid[key].push_back(i);
        }

        // Neighbor cell deltas
        for (const auto& kv : grid) {
            const CellKey& base = kv.first;
            const auto& idxs = kv.second;
            for (int dx=-1; dx<=1; ++dx)
            for (int dy=-1; dy<=1; ++dy)
            for (int dz=-1; dz<=1; ++dz) {
                CellKey nb{base.x+dx, base.y+dy, base.z+dz};
                auto it = grid.find(nb);
                if (it == grid.end()) continue;
                const auto& jdxs = it->second;
                for (std::size_t a = 0; a < idxs.size(); ++a) {
                    for (std::size_t b = 0; b < jdxs.size(); ++b) {
                        std::size_t i = idxs[a];
                        std::size_t j = jdxs[b];
                        if (i>=j) continue;
                        if (seen_pairs.count(pair_id(i,j))) continue;
                        Eigen::Vector3d pi = pos.col(static_cast<Eigen::Index>(i));
                        Eigen::Vector3d pj = pos.col(static_cast<Eigen::Index>(j));
                        if (dist2(pi, pj) <= thresh2) {
                            // bracket refine using next sample
                            // Advance temporary copy by refine_dt for two states
                            StateVector s0i = prop.getSatelliteState(i);
                            StateVector s0j = prop.getSatelliteState(j);
                            // Propagate individual satellites forward by refine_dt using single-sat propagator
                            // Snapshot elements at current time
                            auto ei = prop.getSatelliteElements(i);
                            auto ej = prop.getSatelliteElements(j);
                            OrbitalElements efi{}; efi.a=ei.a; efi.e=ei.e; efi.i=ei.i; efi.O=ei.O; efi.w=ei.w; efi.M=ei.M; efi.t=cur;
                            OrbitalElements efj{}; efj.a=ej.a; efj.e=ej.e; efj.i=ej.i; efj.O=ej.O; efj.w=ej.w; efj.M=ej.M; efj.t=cur;
                            J2OrbitPropagator pi(efi); J2OrbitPropagator pj(efj);
                            pi.setStepSize(std::max(1.0, cfg_.refine_dt/5.0));
                            pj.setStepSize(std::max(1.0, cfg_.refine_dt/5.0));
                            auto efi1 = pi.propagate(cur + cfg_.refine_dt);
                            auto efj1 = pj.propagate(cur + cfg_.refine_dt);
                            StateVector s1i = pi.elementsToState(efi1);
                            StateVector s1j = pj.elementsToState(efj1);
                            auto evs = refineBracket(s0i, s0j, s1i, s1j, cur, std::min(cfg_.horizon, cur+cfg_.refine_dt), cfg_.threshold);
                            for (auto& ev : evs) {
                                ev.sat_i = i; ev.sat_j = j;
                                if (ev.miss_distance <= cfg_.threshold) {
                                    events.push_back(ev);
                                    if (ev.phase == ConjunctionPhase::Closest) {
                                        seen_pairs.insert(pair_id(i,j));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return events;
}

std::vector<ConjunctionEvent> SatelliteConjunctionPredictor::predictHierarchical(J2ConstellationPropagator prop) const {
    std::vector<ConjunctionEvent> events;
    const std::size_t n = prop.getSatelliteCount();
    if (n < 2) return events;
    const double thresh2 = cfg_.threshold * cfg_.threshold;

    double cur = 0.0;
    while (cur < cfg_.horizon) {
        cur = std::min(cfg_.horizon, cur + cfg_.coarse_dt);
        // advance to current coarse time
        prop.propagateConstellation(cur);
        auto pos0 = prop.getAllPositions();
        for (std::size_t i=0;i<n;++i) {
            for (std::size_t j=i+1;j<n;++j) {
                Eigen::Vector3d pi = pos0.col(static_cast<Eigen::Index>(i));
                Eigen::Vector3d pj = pos0.col(static_cast<Eigen::Index>(j));
                if (dist2(pi,pj) <= thresh2) {
                    // refine next dt window
                    StateVector s0i = prop.getSatelliteState(i);
                    StateVector s0j = prop.getSatelliteState(j);
                    auto ei = prop.getSatelliteElements(i);
                    auto ej = prop.getSatelliteElements(j);
                    OrbitalElements efi{}; efi.a=ei.a; efi.e=ei.e; efi.i=ei.i; efi.O=ei.O; efi.w=ei.w; efi.M=ei.M; efi.t=cur;
                    OrbitalElements efj{}; efj.a=ej.a; efj.e=ej.e; efj.i=ej.i; efj.O=ej.O; efj.w=ej.w; efj.M=ej.M; efj.t=cur;
                    J2OrbitPropagator pi(efi); J2OrbitPropagator pj(efj);
                    pi.setStepSize(std::max(1.0, cfg_.refine_dt/5.0));
                    pj.setStepSize(std::max(1.0, cfg_.refine_dt/5.0));
                    auto efi1 = pi.propagate(cur + cfg_.refine_dt);
                    auto efj1 = pj.propagate(cur + cfg_.refine_dt);
                    StateVector s1i = pi.elementsToState(efi1);
                    StateVector s1j = pj.elementsToState(efj1);
                    auto evs = refineBracket(s0i, s0j, s1i, s1j, cur, std::min(cfg_.horizon, cur+cfg_.refine_dt), cfg_.threshold);
                    for (auto& ev : evs) {
                        ev.sat_i = i; ev.sat_j = j;
                        if (ev.miss_distance <= cfg_.threshold) events.push_back(ev);
                    }
                }
            }
        }
    }
    return events;
}

void SatelliteConjunctionPredictor::ensureConnectionTable(const J2ConstellationPropagator& propagator) const {
    const std::size_t n = propagator.getSatelliteCount();
    if (connection_table_valid_ && cached_satellite_count_ == n) return;

    connection_table_.assign(n, {});
    cached_satellite_count_ = n;
    connection_table_valid_ = false;
    if (n < 2 || cfg_.threshold <= 0.0) return;

    std::vector<CompactOrbitalElements> elems(n);
    std::vector<Rates> rates(n);
    for (std::size_t i = 0; i < n; ++i) {
        elems[i] = propagator.getSatelliteElements(i);
        rates[i] = computeRates(elems[i]);
    }

    const double horizon = std::max(0.0, cfg_.horizon);

    struct BucketKey {
        int shell;
        int raan;
        int anomaly;
        bool operator==(const BucketKey& other) const noexcept {
            return shell == other.shell && raan == other.raan && anomaly == other.anomaly;
        }
    };
    struct BucketHash {
        std::size_t operator()(const BucketKey& k) const noexcept {
            std::size_t h1 = std::hash<int>{}(k.shell);
            std::size_t h2 = std::hash<int>{}(k.raan);
            std::size_t h3 = std::hash<int>{}(k.anomaly);
            return h1 ^ (h2 << 1) ^ (h3 << 2);
        }
    };

    double avg_radius = 0.0;
    double avg_sin = 0.0;
    for (const auto& e : elems) {
        double radius = e.a * (1.0 - 0.5 * e.e * e.e);
        avg_radius += radius;
        avg_sin += std::max(1e-3, std::sin(e.i));
    }
    avg_radius = std::max(1.0, avg_radius / static_cast<double>(n));
    avg_sin = std::max(1e-3, avg_sin / static_cast<double>(n));

    const double shell_width = std::max(1000.0, cfg_.threshold);
    double omega_bin_size = std::max(1e-3, cfg_.threshold / (avg_radius * avg_sin));
    double u_bin_size = std::max(1e-3, cfg_.threshold / avg_radius);

    auto bucket_index = [](double angle, double bin_size) -> int {
        double wrapped = wrapPi(angle);
        return static_cast<int>(std::floor((wrapped + M_PI) / bin_size));
    };

    std::unordered_map<BucketKey, std::vector<std::size_t>, BucketHash> buckets;
    buckets.reserve(n * 2);

    std::vector<int> shell_ids(n);
    std::vector<int> raan_ids(n);
    std::vector<int> anomaly_ids(n);

    for (std::size_t i = 0; i < n; ++i) {
        const auto& e = elems[i];
        int shell = static_cast<int>(std::floor(e.a / shell_width));
        int raan = bucket_index(e.O, omega_bin_size);
        int anomaly = bucket_index(e.w + e.M, u_bin_size);
        shell_ids[i] = shell;
        raan_ids[i] = raan;
        anomaly_ids[i] = anomaly;
        buckets[{shell, raan, anomaly}].push_back(i);
    }

    for (std::size_t idx = 0; idx < n; ++idx) {
        const auto& ei = elems[idx];
        const auto& ri = rates[idx];
        int shell = shell_ids[idx];
        int raan = raan_ids[idx];
        int anomaly = anomaly_ids[idx];

        std::unordered_set<std::size_t> candidates;
        for (int ds = -1; ds <= 1; ++ds) {
            int shell_nb = shell + ds;
            for (int dr = -1; dr <= 1; ++dr) {
                int raan_nb = raan + dr;
                for (int da = -1; da <= 1; ++da) {
                    BucketKey key{shell_nb, raan_nb, anomaly + da};
                    auto it = buckets.find(key);
                    if (it == buckets.end()) continue;
                    for (auto id : it->second) {
                        if (id != idx) candidates.insert(id);
                    }
                }
            }
        }

        for (std::size_t j : candidates) {
            if (j <= idx) continue;
            const auto& ej = elems[j];
            const auto& rj = rates[j];

            double a_ref = std::max(1.0, std::min(ei.a, ej.a));
            if (std::abs(ei.a - ej.a) > cfg_.threshold) continue;

            double i_thresh = cfg_.threshold / a_ref;
            if (std::abs(ei.i - ej.i) > i_thresh) continue;

            double sin_i_eff = std::max(1e-3, std::sin(0.5 * (ei.i + ej.i)));
            double omega_thresh = cfg_.threshold / (a_ref * sin_i_eff);
            omega_thresh += std::abs(ri.raandot - rj.raandot) * horizon;
            if (std::abs(wrapPi(ei.O - ej.O)) > omega_thresh) continue;

            double base_u_i = ei.w + ei.M;
            double base_u_j = ej.w + ej.M;
            double u_thresh = cfg_.threshold / a_ref;
            double rate_u_i = ri.argpdot + ri.meandot;
            double rate_u_j = rj.argpdot + rj.meandot;
            u_thresh += std::abs(rate_u_i - rate_u_j) * horizon;
            if (std::abs(wrapPi(base_u_i - base_u_j)) > u_thresh) continue;

            connection_table_[idx].push_back(j);
        }
    }

    connection_table_valid_ = true;
}

std::vector<ConjunctionEvent> SatelliteConjunctionPredictor::predictConnectionTable(J2ConstellationPropagator prop) const {
    std::vector<ConjunctionEvent> events;
    const std::size_t n = prop.getSatelliteCount();
    if (n < 2 || !connection_table_valid_ || connection_table_.size() != n) {
        return events;
    }
    const double thresh2 = cfg_.threshold * cfg_.threshold;

    double cur = 0.0;
    while (cur < cfg_.horizon) {
        cur = std::min(cfg_.horizon, cur + cfg_.coarse_dt);
        prop.propagateConstellation(cur);
        auto pos0 = prop.getAllPositions();
        for (std::size_t i = 0; i < n; ++i) {
            const auto& neighbors = connection_table_[i];
            for (std::size_t j : neighbors) {
                if (j <= i || j >= n) continue;
                Eigen::Vector3d pi = pos0.col(static_cast<Eigen::Index>(i));
                Eigen::Vector3d pj = pos0.col(static_cast<Eigen::Index>(j));
                if (dist2(pi, pj) <= thresh2) {
                    StateVector s0i = prop.getSatelliteState(i);
                    StateVector s0j = prop.getSatelliteState(j);

                    // Refine using single-satellite propagators
                    auto ei = prop.getSatelliteElements(i);
                    auto ej = prop.getSatelliteElements(j);
                    OrbitalElements efi{}; efi.a=ei.a; efi.e=ei.e; efi.i=ei.i; efi.O=ei.O; efi.w=ei.w; efi.M=ei.M; efi.t=cur;
                    OrbitalElements efj{}; efj.a=ej.a; efj.e=ej.e; efj.i=ej.i; efj.O=ej.O; efj.w=ej.w; efj.M=ej.M; efj.t=cur;
                    J2OrbitPropagator pi_single(efi);
                    J2OrbitPropagator pj_single(efj);
                    double refine_step = std::max(1.0, cfg_.refine_dt / 5.0);
                    pi_single.setStepSize(refine_step);
                    pj_single.setStepSize(refine_step);
                    auto efi1 = pi_single.propagate(cur + cfg_.refine_dt);
                    auto efj1 = pj_single.propagate(cur + cfg_.refine_dt);
                    StateVector s1i = pi_single.elementsToState(efi1);
                    StateVector s1j = pj_single.elementsToState(efj1);

                    auto evs = refineBracket(s0i, s0j, s1i, s1j, cur, std::min(cfg_.horizon, cur + cfg_.refine_dt), cfg_.threshold);
                    for (auto& ev : evs) {
                        ev.sat_i = i;
                        ev.sat_j = j;
                        if (ev.miss_distance <= cfg_.threshold) {
                            events.push_back(ev);
                        }
                    }
                }
            }
        }
    }
    return events;
}

void SatelliteConjunctionPredictor::notifyManeuver(std::size_t /*sat_id*/, const CompactOrbitalElements& /*updated_elements*/) const {
    connection_table_valid_ = false;
    cached_satellite_count_ = 0;
}

std::vector<ConjunctionEvent> SatelliteConjunctionPredictor::refineBracket(const StateVector& s0_i, const StateVector& s0_j,
                                                                           const StateVector& s1_i, const StateVector& s1_j,
                                                                           double t0, double t1, double threshold) {
    std::vector<ConjunctionEvent> out;
    const double dt = std::max(1e-6, t1 - t0);
    const double thr2 = threshold * threshold;

    Eigen::Vector3d r0 = s0_i.r - s0_j.r;
    Eigen::Vector3d r1 = s1_i.r - s1_j.r;
    Eigen::Vector3d v = (r1 - r0) / dt;
    const double a = v.squaredNorm();
    const double b = 2.0 * r0.dot(v);
    const double c = r0.squaredNorm() - thr2;

    auto interp_state = [&](const StateVector& s0, const StateVector& s1, double tau)->StateVector{
        double alpha = tau / dt;
        StateVector s{};
        s.r = s0.r + (s1.r - s0.r) * alpha;
        s.v = s0.v + (s1.v - s0.v) * alpha;
        return s;
    };

    // Closest approach within bracket
    double tau_star = 0.0;
    if (a > 0.0) {
        tau_star = - b / (2.0 * a);
        if (tau_star < 0.0) tau_star = 0.0; else if (tau_star > dt) tau_star = dt;
    }
    Eigen::Vector3d r_star = r0 + v * tau_star;
    ConjunctionEvent cca{};
    cca.phase = ConjunctionPhase::Closest;
    cca.time = t0 + tau_star;
    cca.miss_distance = std::sqrt(r_star.squaredNorm());
    cca.state_i = interp_state(s0_i, s1_i, tau_star);
    cca.state_j = interp_state(s0_j, s1_j, tau_star);
    out.push_back(cca);

    // Start/End crossings solving |r0 + v*tau|^2 = thr^2
    if (a > 0.0) {
        double disc = b*b - 4.0*a*c;
        if (disc >= 0.0) {
            double sqrtD = std::sqrt(std::max(0.0, disc));
            double t1r = (-b - sqrtD) / (2.0*a);
            double t2r = (-b + sqrtD) / (2.0*a);
            if (t1r > t2r) std::swap(t1r, t2r);
            double f0 = c; // distance^2 - thr^2 at t0
            // clamp to bracket
            auto in_window = [&](double tau){ return tau >= 0.0 && tau <= dt; };
            if (in_window(t1r)) {
                ConjunctionEvent ev{};
                ev.time = t0 + t1r;
                ev.miss_distance = threshold; // on boundary
                ev.state_i = interp_state(s0_i, s1_i, t1r);
                ev.state_j = interp_state(s0_j, s1_j, t1r);
                ev.phase = (f0 > 0.0) ? ConjunctionPhase::Start : ConjunctionPhase::End;
                out.push_back(ev);
            }
            if (in_window(t2r)) {
                ConjunctionEvent ev{};
                ev.time = t0 + t2r;
                ev.miss_distance = threshold; // on boundary
                ev.state_i = interp_state(s0_i, s1_i, t2r);
                ev.state_j = interp_state(s0_j, s1_j, t2r);
                ev.phase = (f0 > 0.0) ? ConjunctionPhase::End : ConjunctionPhase::Start;
                out.push_back(ev);
            }
        }
    }
    return out;
}
