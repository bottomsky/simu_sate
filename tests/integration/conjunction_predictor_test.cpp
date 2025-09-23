#include <gtest/gtest.h>

#include "constellation_builder.h"
#include "j2_constellation_propagator.h"
#include "SatelliteConjunctionPredictor.h"
#include "math_defs.h"

TEST(ConjunctionPredictorTest, DetectsCloseApproachAtEpoch) {
    // Two satellites nearly co-located along-track in same circular LEO
    WalkerDeltaConfig cfg;
    cfg.plane_count = 1;
    cfg.sats_per_plane = 2;
    cfg.relative_phasing = 0;
    cfg.altitude = 700e3;
    cfg.inclination = 0.0;
    cfg.eccentricity = 0.0;
    cfg.argument_of_perigee = 0.0;
    cfg.raan_offset = 0.0;
    cfg.mean_anomaly_offset = 0.0;
    cfg.epoch = 0.0;

    auto sats = ConstellationBuilder::CreateWalkerDelta(cfg);
    ASSERT_EQ(sats.size(), 2u);

    // Move sat1 by small mean anomaly to be ~500 m apart along-track
    const double r = RE + cfg.altitude;
    const double separation = 500.0; // meters
    sats[1].M = std::fmod(separation / r, 2.0 * M_PI);

    J2ConstellationPropagator prop(cfg.epoch);
    prop.setStepSize(30.0);
    prop.addSatellites(sats);

    ConjunctionPredictorConfig pcfg;
    pcfg.horizon = 600.0;
    pcfg.coarse_dt = 60.0;
    pcfg.refine_dt = 10.0;
    pcfg.threshold = 1000.0;
    pcfg.cell_size = 1000.0;

    SatelliteConjunctionPredictor predictor(pcfg, ConjunctionStrategy::SpatialGrid);
    auto events = predictor.predict(prop);

    // We expect at least one event near t=0 with miss distance ~500 m
    ASSERT_FALSE(events.empty());
    bool found = false;
    for (const auto& ev : events) {
        if ((ev.sat_i == 0 && ev.sat_j == 1) || (ev.sat_i == 1 && ev.sat_j == 0)) {
            EXPECT_NEAR(ev.miss_distance, separation, 50.0); // within 50 m
            found = true;
        }
    }
    EXPECT_TRUE(found);
}
