#include <gtest/gtest.h>

#include "constellation_builder.h"
#include "j2_constellation_propagator.h"
#include "j2_orbit_propagator.h"
#include "math_defs.h"

namespace {
constexpr double kDegreesToRadians(double degrees) {
    return degrees * M_PI / 180.0;
}
}

TEST(BuilderIntegrationTest, WalkerDeltaPropagationMatchesSingleSatellite) {
    WalkerDeltaConfig cfg;
    cfg.plane_count = 3;
    cfg.sats_per_plane = 4;
    cfg.relative_phasing = 1;
    cfg.altitude = 550e3;
    cfg.inclination = kDegreesToRadians(53.0);
    cfg.argument_of_perigee = 0.0;
    cfg.mean_anomaly_offset = 0.0;
    cfg.raan_offset = 0.0;
    cfg.eccentricity = 0.0001;
    cfg.epoch = 0.0;

    auto constellation = ConstellationBuilder::CreateWalkerDelta(cfg);
    ASSERT_EQ(constellation.size(), cfg.plane_count * cfg.sats_per_plane);

    J2ConstellationPropagator propagator(cfg.epoch);
    propagator.setStepSize(60.0);
    propagator.addSatellites(constellation);
    EXPECT_EQ(propagator.getSatelliteCount(), constellation.size());

    const double target_time = 900.0;  // 15 minutes
    propagator.propagateConstellation(target_time);

    auto positions = propagator.getAllPositions();
    ASSERT_EQ(positions.cols(), constellation.size());
    ASSERT_EQ(positions.rows(), 3);

    // Validate the first satellite against a standalone propagator
    const auto& elem0 = constellation.front();
    OrbitalElements initial{};
    initial.a = elem0.a;
    initial.e = elem0.e;
    initial.i = elem0.i;
    initial.O = elem0.O;
    initial.w = elem0.w;
    initial.M = elem0.M;
    initial.t = cfg.epoch;

    J2OrbitPropagator reference(initial);
    reference.setStepSize(60.0);
    auto propagated_elements = reference.propagate(target_time);
    auto reference_state = reference.elementsToState(propagated_elements);

    StateVector constellation_state = propagator.getSatelliteState(0);

    constexpr double kPositionTolerance = 5e-2;  // 5 cm
    constexpr double kVelocityTolerance = 1e-4;  // 0.1 mm/s

    EXPECT_NEAR(constellation_state.r.x(), reference_state.r.x(), kPositionTolerance);
    EXPECT_NEAR(constellation_state.r.y(), reference_state.r.y(), kPositionTolerance);
    EXPECT_NEAR(constellation_state.r.z(), reference_state.r.z(), kPositionTolerance);

    EXPECT_NEAR(constellation_state.v.x(), reference_state.v.x(), kVelocityTolerance);
    EXPECT_NEAR(constellation_state.v.y(), reference_state.v.y(), kVelocityTolerance);
    EXPECT_NEAR(constellation_state.v.z(), reference_state.v.z(), kVelocityTolerance);
}

TEST(BuilderIntegrationTest, WalkerDeltaCoarseStepPropagation) {
    WalkerDeltaConfig cfg;
    cfg.plane_count = 2;
    cfg.sats_per_plane = 3;
    cfg.relative_phasing = 1;
    cfg.altitude = 550e3;
    cfg.inclination = kDegreesToRadians(53.0);
    cfg.argument_of_perigee = 0.0;
    cfg.mean_anomaly_offset = 0.0;
    cfg.raan_offset = 0.0;
    cfg.eccentricity = 0.0001;
    cfg.epoch = 0.0;

    auto constellation = ConstellationBuilder::CreateWalkerDelta(cfg);
    ASSERT_EQ(constellation.size(), cfg.plane_count * cfg.sats_per_plane);

    const double target_time = 3600.0;  // 1 hour

    J2ConstellationPropagator fine(cfg.epoch);
    fine.setStepSize(30.0);
    fine.addSatellites(constellation);
    fine.propagateConstellation(target_time);

    J2ConstellationPropagator coarse(cfg.epoch);
    coarse.setStepSize(30.0);
    coarse.addSatellites(constellation);
    coarse.propagateConstellationWithStep(target_time, 300.0);

    StateVector fine_state = fine.getSatelliteState(0);
    StateVector coarse_state = coarse.getSatelliteState(0);

    constexpr double kCoarsePositionTolerance = 5e3;   // 5 km
    constexpr double kCoarseVelocityTolerance = 5.0;   // 5 m/s

    EXPECT_NEAR(coarse_state.r.x(), fine_state.r.x(), kCoarsePositionTolerance);
    EXPECT_NEAR(coarse_state.r.y(), fine_state.r.y(), kCoarsePositionTolerance);
    EXPECT_NEAR(coarse_state.r.z(), fine_state.r.z(), kCoarsePositionTolerance);

    EXPECT_NEAR(coarse_state.v.x(), fine_state.v.x(), kCoarseVelocityTolerance);
    EXPECT_NEAR(coarse_state.v.y(), fine_state.v.y(), kCoarseVelocityTolerance);
    EXPECT_NEAR(coarse_state.v.z(), fine_state.v.z(), kCoarseVelocityTolerance);

}
