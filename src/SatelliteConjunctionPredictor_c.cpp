#include "SatelliteConjunctionPredictor_c.h"
#include "SatelliteConjunctionPredictor.h"
#include "j2_constellation_propagator.h"
#include <vector>

struct CConjunctionPredictor {
    ConjunctionPredictorConfig cfg;
    SatelliteConjunctionPredictor impl;
    std::vector<ConjunctionEvent> last_events;

    explicit CConjunctionPredictor(const ConjunctionPredictorConfig& c)
        : cfg(c), impl(cfg, ConjunctionStrategy::SpatialGrid) {}
};

static ConjunctionPredictorConfig to_cpp_config(const CConjunctionPredictorConfig* c) {
    ConjunctionPredictorConfig out{};
    if (c) {
        out.horizon = c->horizon_s;
        out.coarse_dt = c->coarse_dt_s;
        out.refine_dt = c->refine_dt_s;
        out.threshold = c->threshold_m;
        out.cell_size = c->cell_size_m;
    }
    return out;
}

static CConjunctionPhase to_c_phase(ConjunctionPhase p) {
    switch (p) {
        case ConjunctionPhase::Start:   return CONJUNCTION_PHASE_START;
        case ConjunctionPhase::Closest: return CONJUNCTION_PHASE_CLOSEST;
        case ConjunctionPhase::End:     return CONJUNCTION_PHASE_END;
        default:                        return CONJUNCTION_PHASE_CLOSEST;
    }
}

extern "C" {

J2_API CConjunctionPredictor* j2_conjunction_predictor_create(const CConjunctionPredictorConfig* config) {
    try {
        ConjunctionPredictorConfig cpp_cfg = to_cpp_config(config);
        return new CConjunctionPredictor(cpp_cfg);
    } catch (...) {
        return nullptr;
    }
}

J2_API void j2_conjunction_predictor_destroy(CConjunctionPredictor* predictor) {
    delete predictor;
}

J2_API J2Status j2_conjunction_predictor_predict(CConjunctionPredictor* predictor,
                                                 J2ConstellationPropagatorHandle propagator) {
    if (!predictor || !propagator) return J2_STATUS_INVALID_ARGUMENT;
    try {
        auto* prop = static_cast<J2ConstellationPropagator*>(propagator);
        predictor->last_events = predictor->impl.predict(*prop);
        return J2_STATUS_SUCCESS;
    } catch (...) {
        return J2_STATUS_INTERNAL_ERROR;
    }
}

J2_API int j2_conjunction_predictor_get_event_count(const CConjunctionPredictor* predictor) {
    if (!predictor) return -1;
    return static_cast<int>(predictor->last_events.size());
}

J2_API J2Status j2_conjunction_predictor_get_events(const CConjunctionPredictor* predictor,
                                                    CConjunctionEvent* events,
                                                    int max_count,
                                                    int* actual_count) {
    if (!predictor || !events || !actual_count || max_count <= 0) return J2_STATUS_INVALID_ARGUMENT;
    try {
        const auto& evs = predictor->last_events;
        const int n = static_cast<int>(evs.size());
        const int m = (n > max_count) ? max_count : n;
        *actual_count = m;
        for (int i = 0; i < m; ++i) {
            const auto& e = evs[static_cast<std::size_t>(i)];
            events[i].satellite1_id = static_cast<int>(e.sat_i);
            events[i].satellite2_id = static_cast<int>(e.sat_j);
            events[i].phase = to_c_phase(e.phase);
            events[i].time_s = e.time;
            events[i].miss_distance_m = e.miss_distance;
            // 拷贝状态向量
            events[i].state_i.r[0] = e.state_i.r.x();
            events[i].state_i.r[1] = e.state_i.r.y();
            events[i].state_i.r[2] = e.state_i.r.z();
            events[i].state_i.v[0] = e.state_i.v.x();
            events[i].state_i.v[1] = e.state_i.v.y();
            events[i].state_i.v[2] = e.state_i.v.z();
            events[i].state_j.r[0] = e.state_j.r.x();
            events[i].state_j.r[1] = e.state_j.r.y();
            events[i].state_j.r[2] = e.state_j.r.z();
            events[i].state_j.v[0] = e.state_j.v.x();
            events[i].state_j.v[1] = e.state_j.v.y();
            events[i].state_j.v[2] = e.state_j.v.z();
        }
        return J2_STATUS_SUCCESS;
    } catch (...) {
        return J2_STATUS_INTERNAL_ERROR;
    }
}

J2_API J2Status j2_conjunction_predictor_clear(CConjunctionPredictor* predictor) {
    if (!predictor) return J2_STATUS_INVALID_ARGUMENT;
    predictor->last_events.clear();
    return J2_STATUS_SUCCESS;
}

} // extern "C"
