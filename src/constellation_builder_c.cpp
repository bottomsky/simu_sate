#include "constellation_builder_c.h"
#include "constellation_builder.h"
#include "math_defs.h"
#include <vector>
#include <cmath>
#include <stdexcept>

extern "C" {

static bool validate_c_config(const CWalkerDeltaConfig& c) {
    if (c.plane_count <= 0 || c.satellites_per_plane <= 0) {
        return false;
    }
    if (c.relative_phasing < 0) {
        return false;
    }
    if (!std::isfinite(c.altitude_m) || !std::isfinite(c.inclination_rad) ||
        !std::isfinite(c.eccentricity) || !std::isfinite(c.argument_of_perigee_rad) ||
        !std::isfinite(c.mean_anomaly_offset_rad) || !std::isfinite(c.raan_offset_rad) ||
        !std::isfinite(c.epoch_seconds)) {
        return false;
    }
    if (c.eccentricity < 0.0 || c.eccentricity >= 1.0) {
        return false;
    }
    if (c.inclination_rad < 0.0 || c.inclination_rad > M_PI) {
        return false;
    }
    if (c.altitude_m + RE <= 0.0) {
        return false;
    }
    return true;
}

static WalkerDeltaConfig to_cpp_config(const CWalkerDeltaConfig& c) {
    WalkerDeltaConfig cfg{};
    cfg.plane_count = static_cast<std::size_t>(c.plane_count);
    cfg.sats_per_plane = static_cast<std::size_t>(c.satellites_per_plane);
    cfg.relative_phasing = static_cast<std::size_t>(c.relative_phasing);
    cfg.altitude = c.altitude_m;
    cfg.inclination = c.inclination_rad;
    cfg.eccentricity = c.eccentricity;
    cfg.argument_of_perigee = normalizeAngle(c.argument_of_perigee_rad);
    cfg.mean_anomaly_offset = normalizeAngle(c.mean_anomaly_offset_rad);
    cfg.raan_offset = normalizeAngle(c.raan_offset_rad);
    cfg.epoch = c.epoch_seconds;
    return cfg;
}

static void to_c_elements(const CompactOrbitalElements& in, CCompactOrbitalElements* out) {
    out->a = in.a;
    out->e = in.e;
    out->i = in.i;
    out->O = in.O;
    out->w = in.w;
    out->M = in.M;
}

int j2_constellation_walker_delta_count(const CWalkerDeltaConfig* config) {
    if (!config) return -1;
    if (config->plane_count <= 0 || config->satellites_per_plane <= 0) return -1;
    return config->plane_count * config->satellites_per_plane;
}

J2Status j2_constellation_create_walker_delta(
    const CWalkerDeltaConfig* config,
    CCompactOrbitalElements* elements,
    int max_count,
    int* actual_count
) {
    if (!config || !elements || !actual_count) return J2_STATUS_INVALID_ARGUMENT;
    if (max_count <= 0) return J2_STATUS_INVALID_ARGUMENT;
    if (!validate_c_config(*config)) return J2_STATUS_INVALID_ARGUMENT;

    try {
        WalkerDeltaConfig cpp_cfg = to_cpp_config(*config);
        std::vector<CompactOrbitalElements> out = ConstellationBuilder::CreateWalkerDelta(cpp_cfg);
        const int n = static_cast<int>(out.size());
        const int m = (n > max_count) ? max_count : n;
        *actual_count = m;
        for (int i = 0; i < m; ++i) {
            to_c_elements(out[static_cast<std::size_t>(i)], &elements[i]);
        }
        return J2_STATUS_SUCCESS;
    } catch (...) {
        return J2_STATUS_INTERNAL_ERROR;
    }
}

} // extern "C"
