#include "CoordinateConverter.h"

#include <algorithm>
#include <cmath>

#include "math_defs.h"

double CoordinateConverter::normalizeAngle(double angle) {
    return ::normalizeAngle(angle);
}

double CoordinateConverter::computeEccentricAnomaly(double M, double e) {
    M = normalizeAngle(M);
    double E = (e < 0.8) ? M : (M > M_PI ? M - e : M + e);

    for (int iter = 0; iter < 20; ++iter) {
        double delta = (E - e * std::sin(E) - M) / (1.0 - e * std::cos(E));
        E -= delta;
        if (std::abs(delta) < EPSILON) {
            break;
        }
    }

    return E;
}

double CoordinateConverter::computeTrueAnomaly(double E, double e) {
    double tan_half_nu = std::sqrt((1.0 + e) / (1.0 - e)) * std::tan(E / 2.0);
    double nu = 2.0 * std::atan(tan_half_nu);
    return normalizeAngle(nu);
}

Eigen::Matrix3d CoordinateConverter::perifocalToEci(double O, double i, double w) {
    Eigen::Matrix3d R;
    double cosO = std::cos(O), sinO = std::sin(O);
    double cosi = std::cos(i), sini = std::sin(i);
    double cosw = std::cos(w), sinw = std::sin(w);

    R(0,0) = cosO * cosw - sinO * sinw * cosi;
    R(0,1) = -cosO * sinw - sinO * cosw * cosi;
    R(0,2) = sinO * sini;

    R(1,0) = sinO * cosw + cosO * sinw * cosi;
    R(1,1) = -sinO * sinw + cosO * cosw * cosi;
    R(1,2) = -cosO * sini;

    R(2,0) = sinw * sini;
    R(2,1) = cosw * sini;
    R(2,2) = cosi;

    return R;
}

StateVector CoordinateConverter::elementsToStateImpl(double a, double e, double i, double O, double w, double M) {
    StateVector state;

    double E = computeEccentricAnomaly(M, e);
    double nu = computeTrueAnomaly(E, e);

    double r_mag = a * (1.0 - e * std::cos(E));

    Eigen::Vector3d r_perifocal(r_mag * std::cos(nu), r_mag * std::sin(nu), 0.0);

    double p = a * (1.0 - e * e);
    double v_mag_factor = std::sqrt(MU / p);
    Eigen::Vector3d v_perifocal(-v_mag_factor * std::sin(nu),
                                v_mag_factor * (e + std::cos(nu)),
                                0.0);

    Eigen::Matrix3d R = perifocalToEci(O, i, w);
    state.r = R * r_perifocal;
    state.v = R * v_perifocal;

    return state;
}

StateVector CoordinateConverter::elementsToState(const OrbitalElements& elements) {
    return elementsToStateImpl(elements.a, elements.e, elements.i,
                               elements.O, elements.w, elements.M);
}

StateVector CoordinateConverter::elementsToState(const CompactOrbitalElements& elements) {
    return elementsToStateImpl(elements.a, elements.e, elements.i,
                               elements.O, elements.w, elements.M);
}

OrbitalElements CoordinateConverter::stateToElements(const StateVector& state, double epoch) {
    OrbitalElements elements{};
    elements.t = epoch;

    const Eigen::Vector3d& r_vec = state.r;
    const Eigen::Vector3d& v_vec = state.v;
    double r = r_vec.norm();
    double v = v_vec.norm();

    Eigen::Vector3d h_vec = r_vec.cross(v_vec);
    double h = h_vec.norm();

    Eigen::Vector3d k_hat(0.0, 0.0, 1.0);
    Eigen::Vector3d n_vec = k_hat.cross(h_vec);
    double n = n_vec.norm();

    Eigen::Vector3d e_vec = ((v * v - MU / r) * r_vec - (r_vec.dot(v_vec)) * v_vec) / MU;
    double ecc = e_vec.norm();
    elements.e = ecc;

    double energy = v * v / 2.0 - MU / r;
    elements.a = -MU / (2.0 * energy);

    constexpr double SMALL_ECC = 1e-8;
    constexpr double SMALL_NORM = 1e-12;

    double cos_i = std::clamp(h_vec.z() / h, -1.0, 1.0);
    elements.i = std::acos(cos_i);

    bool equatorial = n < SMALL_NORM;
    if (!equatorial) {
        double cos_O = std::clamp(n_vec.x() / n, -1.0, 1.0);
        elements.O = std::acos(cos_O);
        if (n_vec.y() < 0) {
            elements.O = 2.0 * M_PI - elements.O;
        }
    } else {
        elements.O = 0.0;
    }

    double argument_of_perigee = 0.0;
    double true_anomaly = 0.0;

    if (ecc > SMALL_ECC) {
        if (!equatorial) {
            double cos_w = std::clamp(n_vec.dot(e_vec) / (n * ecc), -1.0, 1.0);
            argument_of_perigee = std::acos(cos_w);
            if (e_vec.z() < 0) {
                argument_of_perigee = 2.0 * M_PI - argument_of_perigee;
            }
        } else {
            argument_of_perigee = normalizeAngle(std::atan2(e_vec.y(), e_vec.x()));
        }

        double cos_nu = std::clamp(e_vec.dot(r_vec) / (ecc * r), -1.0, 1.0);
        true_anomaly = std::acos(cos_nu);
        if (r_vec.dot(v_vec) < 0) {
            true_anomaly = 2.0 * M_PI - true_anomaly;
        }
    } else {
        elements.e = 0.0;
        Eigen::Vector3d h_hat = h_vec.normalized();
        Eigen::Vector3d node_hat;
        if (!equatorial) {
            node_hat = n_vec / n;
        } else {
            node_hat = Eigen::Vector3d::UnitX();
        }
        Eigen::Vector3d perigee_hat = h_hat.cross(node_hat);
        true_anomaly = std::atan2(r_vec.dot(perigee_hat), r_vec.dot(node_hat));
        true_anomaly = normalizeAngle(true_anomaly);
    }

    elements.w = normalizeAngle(argument_of_perigee);

    double mean_anomaly = true_anomaly;
    if (elements.e > SMALL_ECC) {
        double sqrt_one_minus_e2 = std::sqrt(std::max(0.0, 1.0 - elements.e * elements.e));
        double sin_E = sqrt_one_minus_e2 * std::sin(true_anomaly) / (1.0 + elements.e * std::cos(true_anomaly));
        double cos_E = (elements.e + std::cos(true_anomaly)) / (1.0 + elements.e * std::cos(true_anomaly));
        double E = std::atan2(sin_E, cos_E);
        E = normalizeAngle(E);
        mean_anomaly = E - elements.e * std::sin(E);
    }

    elements.M = normalizeAngle(mean_anomaly);
    elements.i = normalizeAngle(elements.i);
    elements.O = normalizeAngle(elements.O);
    elements.w = normalizeAngle(elements.w);
    elements.M = normalizeAngle(elements.M);

    return elements;
}

CompactOrbitalElements CoordinateConverter::stateToCompactElements(const StateVector& state, double epoch) {
    OrbitalElements full = stateToElements(state, epoch);
    CompactOrbitalElements compact{};
    compact.a = full.a;
    compact.e = full.e;
    compact.i = full.i;
    compact.O = full.O;
    compact.w = full.w;
    compact.M = full.M;
    return compact;
}
