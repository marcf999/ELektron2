#pragma once

#include <cmath>

namespace PhysicalData {

    // Impact parameter range in meters
    constexpr double rangeMin = 1e-12;
    constexpr double rangeMax = 1e-10;

    constexpr int spin = +1;

    // Zitter radius in m: hbar/(2mc)
    constexpr double zitterRadius = 1.93079634e-13;

    // Simulation parameters
    constexpr double startEnergy = 5000.0;       // eV
    constexpr double startPos = -1000.0;          // reduced units (zitter radii)
    constexpr int totalSimulations = 1000;
    constexpr int plotsToShow = 10;

    // Integrator tolerances
    constexpr double relTol = 1e-12;
    constexpr double absTol = 1e-12;
    constexpr double minStep = 1e-10;
    constexpr double maxStep = 10.0;

    // Detection cutoff in reduced units
    constexpr double detectionDistance = 1000.0;

    // Max integration time (reduced units)
    constexpr double maxTime = 1e6;

    // Logging
    constexpr bool debug = true;
    constexpr bool logInitialConditions = false;
    constexpr int progressLogEvery = 100;

    // Integrity tolerances
    constexpr double radiusTolerance = 1e-1;
    constexpr double zdot2Tolerance = 1e-1;

    // Dirac frequency in s^-1
    constexpr double zetaFrequency = 1.55268814e21;

    // Dirac Time in s
    constexpr double ZitterTime = 6.4404433e-22;

    // Electron Rest mass in kilograms
    constexpr double electronRestMass = 9.109e-31;

    // From notes of Rivas natural units A = (ehbar)/(2m2c3)
    constexpr double rivasA = 3.778e-19;
    constexpr double coulombFactor = 8.988e+9;
    constexpr double electronCharge = -1.60217663e-19;
    constexpr double protonCharge = -electronCharge;

    // Carbon nucleus: atomic number Z = 6
    constexpr double carbonProtons = 6.0;

    // Velocity of light
    constexpr double c = 2.99792458e+8;
    constexpr double c2 = c * c;

    // Electron rest energy in eV
    constexpr double m0c2 = 5.11e+5;

    // Fine structure constant
    constexpr double alpha = 0.007299270072992700;

    // Bohr Radius in meters for screening
    constexpr double bohrRadius = 5.3e-11;

    // Reduced Bohr radius (in zitter radii)
    inline const double reducedBohr = bohrRadius / zitterRadius; // ~275

} // namespace PhysicalData
