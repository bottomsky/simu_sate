#ifndef CONSTELLATION_BUILDER_H
#define CONSTELLATION_BUILDER_H

#include <vector>
#include <cstddef>

#include "common_types.h"
#include "j2_constellation_propagator.h"

struct WalkerDeltaConfig {
    std::size_t plane_count = 1;       // Number of orbital planes (A)
    std::size_t sats_per_plane = 1;    // Satellites per plane (B)
    std::size_t relative_phasing = 0;  // Walker F parameter
    double altitude = 550e3;           // Altitude above Earth (m)
    double inclination = 0.0;          // Inclination (rad)
    double eccentricity = 0.0;         // Orbit eccentricity
    double argument_of_perigee = 0.0;  // Argument of perigee (rad)
    double mean_anomaly_offset = 0.0;  // Global mean anomaly offset (rad)
    double raan_offset = 0.0;          // Reference RAAN (rad)
    double epoch = 0.0;                // Epoch time (s)
};

class ConstellationBuilder {
public:
    static std::vector<CompactOrbitalElements> CreateWalkerDelta(const WalkerDeltaConfig& cfg);
};

#endif // CONSTELLATION_BUILDER_H
