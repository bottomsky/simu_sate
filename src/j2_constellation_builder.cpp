#include "j2_constellation_builder.h"

#include <cmath>
#include <stdexcept>

#include "math_defs.h"

namespace {
constexpr double kTwoPi = 2.0 * M_PI;

void validate_config(const WalkerDeltaConfig& cfg) {
    if (cfg.plane_count == 0) {
        throw std::invalid_argument("plane_count must be greater than zero");
    }
    if (cfg.sats_per_plane == 0) {
        throw std::invalid_argument("sats_per_plane must be greater than zero");
    }
    if (cfg.eccentricity < 0.0 || cfg.eccentricity >= 1.0) {
        throw std::invalid_argument("eccentricity must be within [0, 1)");
    }
    if (cfg.altitude + RE <= 0.0) {
        throw std::invalid_argument("semi-major axis must be positive");
    }
}
}

std::vector<CompactOrbitalElements> J2ConstellationBuilder::CreateWalkerDelta(const WalkerDeltaConfig& cfg) {
    validate_config(cfg);

    const std::size_t total_satellites = cfg.plane_count * cfg.sats_per_plane;
    const double semi_major_axis = RE + cfg.altitude;
    const double anomaly_spacing = kTwoPi / static_cast<double>(cfg.sats_per_plane);
    const double raan_spacing = kTwoPi / static_cast<double>(cfg.plane_count);
    const double phasing_term = kTwoPi * static_cast<double>(cfg.relative_phasing) /
                                static_cast<double>(total_satellites);

    std::vector<CompactOrbitalElements> constellation;
    constellation.reserve(total_satellites);

    for (std::size_t plane_idx = 0; plane_idx < cfg.plane_count; ++plane_idx) {
        const double raan = cfg.raan_offset + raan_spacing * static_cast<double>(plane_idx);
        const double plane_phase = phasing_term * static_cast<double>(plane_idx);

        for (std::size_t sat_idx = 0; sat_idx < cfg.sats_per_plane; ++sat_idx) {
            CompactOrbitalElements elements{};
            elements.a = semi_major_axis;
            elements.e = cfg.eccentricity;
            elements.i = cfg.inclination;
            elements.O = std::fmod(raan, kTwoPi);
            if (elements.O < 0.0) elements.O += kTwoPi;
            elements.w = cfg.argument_of_perigee;

            double mean_anomaly = cfg.mean_anomaly_offset + anomaly_spacing * static_cast<double>(sat_idx) + plane_phase;
            mean_anomaly = std::fmod(mean_anomaly, kTwoPi);
            if (mean_anomaly < 0.0) mean_anomaly += kTwoPi;
            elements.M = mean_anomaly;

            constellation.push_back(elements);
        }
    }

    return constellation;
}
