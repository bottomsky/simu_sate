#ifndef SATELLITE_CONJUNCTION_PREDICTOR_H
#define SATELLITE_CONJUNCTION_PREDICTOR_H

#include <cstddef>
#include <vector>
#include <utility>

#include "common_types.h"
#include "j2_constellation_propagator.h"

enum class ConjunctionPhase { Start, Closest, End };

struct ConjunctionEvent {
    std::size_t sat_i;
    std::size_t sat_j;
    ConjunctionPhase phase; // Start/Closest/End
    double time;            // event time (s, relative to predictor epoch)
    double miss_distance;   // separation at event time (m)
    StateVector state_i;    // ECI position/velocity of sat_i at event time
    StateVector state_j;    // ECI position/velocity of sat_j at event time
};

struct ConjunctionPredictorConfig {
    double horizon = 3600.0;      // prediction window (s)
    double coarse_dt = 60.0;      // coarse sampling step (s)
    double refine_dt = 5.0;       // refinement sampling step (s)
    double threshold = 1000.0;    // detection distance (m)
    double cell_size = 1000.0;    // spatial grid cell size (m), typically = threshold
};

enum class ConjunctionStrategy {
    Auto,
    Hierarchical,   // coarse screen + local refinement
    SpatialGrid,    // spatial hashing per snapshot
    ConnectionTable // orbit-element neighbor prefilter (optional)
};

class SatelliteConjunctionPredictor {
public:
    SatelliteConjunctionPredictor(ConjunctionPredictorConfig cfg,
                                  ConjunctionStrategy strat = ConjunctionStrategy::Auto)
        : cfg_(cfg), strategy_(strat) {}

    // Predict conjunctions within [0, horizon]
    std::vector<ConjunctionEvent> predict(const J2ConstellationPropagator& propagator) const;

private:
    ConjunctionPredictorConfig cfg_;
    ConjunctionStrategy strategy_;

    // strategies
    std::vector<ConjunctionEvent> predictSpatialGrid(J2ConstellationPropagator prop) const;
    std::vector<ConjunctionEvent> predictHierarchical(J2ConstellationPropagator prop) const;
    std::vector<ConjunctionEvent> predictConnectionTable(J2ConstellationPropagator prop) const;

    // utilities
    static std::vector<ConjunctionEvent> refineBracket(const StateVector& s0_i, const StateVector& s0_j,
                                                       const StateVector& s1_i, const StateVector& s1_j,
                                                       double t0, double t1, double threshold);

    void ensureConnectionTable(const J2ConstellationPropagator& propagator) const;

    mutable std::vector<std::vector<std::size_t>> connection_table_;
    mutable std::size_t cached_satellite_count_ = 0;
    mutable bool connection_table_valid_ = false;
};

#endif // SATELLITE_CONJUNCTION_PREDICTOR_H
