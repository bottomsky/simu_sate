#ifndef ORBITAL_ELEMENTS_H
#define ORBITAL_ELEMENTS_H

#include <cstddef>

struct OrbitalElements {
    double a;
    double e;
    double i;
    double O;
    double w;
    double M;
    double t;
};

struct CompactOrbitalElements {
    double a;
    double e;
    double i;
    double O;
    double w;
    double M;
};

#endif // ORBITAL_ELEMENTS_H
