#ifndef COORDINATE_CONVERTER_H
#define COORDINATE_CONVERTER_H

#include <Eigen/Dense>
#include "common_types.h"
#include "orbital_elements.h"

class CoordinateConverter {
public:
    static double normalizeAngle(double angle);
    static double computeEccentricAnomaly(double M, double e);
    static double computeTrueAnomaly(double E, double e);

    static StateVector elementsToState(const OrbitalElements& elements);
    static StateVector elementsToState(const CompactOrbitalElements& elements);

    static OrbitalElements stateToElements(const StateVector& state, double epoch);
    static CompactOrbitalElements stateToCompactElements(const StateVector& state, double epoch);

private:
    static Eigen::Matrix3d perifocalToEci(double O, double i, double w);
    static StateVector elementsToStateImpl(double a, double e, double i, double O, double w, double M);
};

#endif // COORDINATE_CONVERTER_H
