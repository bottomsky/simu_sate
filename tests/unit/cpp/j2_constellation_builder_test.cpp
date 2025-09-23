#include <gtest/gtest.h>

#include "j2_constellation_builder.h"

namespace {
constexpr double kTwoPi = 2.0 * M_PI;

double normalize(double value) {
    double x = std::fmod(value, kTwoPi);
    if (x < 0.0) x += kTwoPi;
    return x;
}
}

TEST(J2ConstellationBuilderTest, WalkerDeltaSpacing) {
    WalkerDeltaConfig cfg;
    cfg.plane_count = 3;
    cfg.sats_per_plane = 4;
    cfg.relative_phasing = 1;
    cfg.altitude = 550e3;
    cfg.inclination = 53.0 * M_PI / 180.0;

    auto constellation = J2ConstellationBuilder::CreateWalkerDelta(cfg);
    ASSERT_EQ(constellation.size(), cfg.plane_count * cfg.sats_per_plane);

    const double expected_raan_spacing = kTwoPi / static_cast<double>(cfg.plane_count);
    const double expected_anomaly_spacing = kTwoPi / static_cast<double>(cfg.sats_per_plane);
    const double phasing_term = kTwoPi * static_cast<double>(cfg.relative_phasing) /
                                static_cast<double>(cfg.plane_count * cfg.sats_per_plane);

    for (std::size_t plane = 0; plane < cfg.plane_count; ++plane) {
        double expected_raan = normalize(cfg.raan_offset + expected_raan_spacing * static_cast<double>(plane));
        for (std::size_t slot = 0; slot < cfg.sats_per_plane; ++slot) {
            const auto& elem = constellation[plane * cfg.sats_per_plane + slot];
            EXPECT_NEAR(elem.a, RE + cfg.altitude, 1e-6);
            EXPECT_NEAR(elem.e, cfg.eccentricity, 1e-12);
            EXPECT_NEAR(elem.i, cfg.inclination, 1e-12);
            EXPECT_NEAR(elem.O, expected_raan, 1e-12);

            double expected_M = normalize(cfg.mean_anomaly_offset + expected_anomaly_spacing * static_cast<double>(slot) +
                                          phasing_term * static_cast<double>(plane));
            EXPECT_NEAR(elem.M, expected_M, 1e-12);
        }
    }
}
