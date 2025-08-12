#include <gtest/gtest.h>
#include "j2_orbit_propagator.h"
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// Helper function to write orbital elements to a JSON file
void write_orbit_data_to_json(const std::string& filename, const OrbitalElements& initial_elements, const OrbitalElements& propagated_elements, double step_size) {
    json j;
    j["step_size_seconds"] = step_size;
    j["initial_elements"] = {
        {"epoch", initial_elements.t},
        {"a", initial_elements.a},
        {"e", initial_elements.e},
        {"i", initial_elements.i},
        {"O", initial_elements.O},
        {"w", initial_elements.w},
        {"M", initial_elements.M}
    };
    j["propagated_elements"] = {
        {"epoch", propagated_elements.t},
        {"a", propagated_elements.a},
        {"e", propagated_elements.e},
        {"i", propagated_elements.i},
        {"O", propagated_elements.O},
        {"w", propagated_elements.w},
        {"M", propagated_elements.M}
    };

    std::ofstream o(filename);
    o << std::setw(4) << j << std::endl;
}

TEST(SingleSatellitePropagation, PropagateAndRecord) {
    // Initial orbital elements for a test satellite (e.g., LEO)
    OrbitalElements initial_elements;
    initial_elements.t = 0.0; // Epoch time in seconds
    initial_elements.a = 7000000.0; // Semi-major axis in meters
    initial_elements.e = 0.001;      // Eccentricity
    initial_elements.i = 1.0;      // Inclination in radians
    initial_elements.O = 0.5;      // RAAN in radians
    initial_elements.w = 0.2;      // Argument of perigee in radians
    initial_elements.M = 0.0;        // Mean anomaly in radians

    double propagation_duration = 86400.0; // 1 day

    std::map<std::string, double> step_sizes;
    step_sizes["1_sec"] = 1.0;
    step_sizes["1_min"] = 60.0;
    step_sizes["1_hour"] = 3600.0;
    step_sizes["1_day"] = 86400.0;

    for (const auto& pair : step_sizes) {
        std::string step_name = pair.first;
        double step_size = pair.second;

        J2OrbitPropagator propagator(initial_elements);
        propagator.setStepSize(step_size);

        OrbitalElements propagated_elements = propagator.propagate(initial_elements.t + propagation_duration);

        std::string filename = "single_sat_propagation_" + step_name + ".json";
        write_orbit_data_to_json(filename, initial_elements, propagated_elements, step_size);

        // Basic assertion to check if propagation happened
        ASSERT_NE(propagated_elements.M, initial_elements.M);
    }
}