#pragma once

#include <cmath>
#include <array>

namespace PhysicalData {

    // Integrator choice: CAPD, Boost, or DP853
    enum class Integrator { CAPD, Boost, DP853 };
    constexpr Integrator integrator = Integrator::DP853;

    // Impact parameter range in meters
    constexpr double rangeMin = 1e-12;
    constexpr double rangeMax = 1e-10;

    constexpr int spin = +1;

    // Zitter radius in m: hbar/(2mc)
    constexpr double zitterRadius = 1.93079634e-13;

    // Atom chain: 30 carbon atoms along z-axis, C-C bond length spacing (graphene)
    constexpr int atomCount = 30;
    constexpr double atomSpacingMeters = 1.42e-10;  // graphene C-C bond length in meters
    constexpr double atomSpacing = atomSpacingMeters / zitterRadius;  // ~735.5 reduced units
    constexpr double chainHalfLength = (atomCount - 1) / 2.0 * atomSpacing;

    // Pre-computed atom z-positions in reduced units, centered at z=0
    inline const std::array<double, atomCount> atomZ = []() {
        std::array<double, atomCount> pos{};
        double sp = atomSpacingMeters / zitterRadius;
        for (int i = 0; i < atomCount; i++) {
            pos[i] = (i - (atomCount - 1) / 2.0) * sp;
        }
        return pos;
    }();

    // Simulation parameters
    constexpr double startEnergy = 5000.0;       // eV
    constexpr double startPos = -(chainHalfLength + 4000.0);  // reduced units, well before first atom
    constexpr int totalSimulations = 100;
    constexpr int plotsToShow = 0;

    // CAPD Taylor integrator parameters
    constexpr int capdOrder = 20;
    constexpr double capdStepDivisor = 10.0;
    constexpr double capdMinStep = 1e-4;
    constexpr double maxStep = 50.0;

    // Boost.Odeint DormandPrince5(4) parameters
    constexpr double boostAbsTol = 1e-12;
    constexpr double boostRelTol = 1e-12;

    // DP853 integrator parameters (matches Java commons-math3)
    constexpr double dp853AbsTol = 1e-12;
    constexpr double dp853RelTol = 1e-12;
    constexpr double dp853MinStep = 1e-10;
    constexpr double dp853MaxStep = 10.0;

    // Detection cutoff in reduced units — beyond the atom chain
    constexpr double detectionDistance = chainHalfLength + 4000.0;

    // Max integration time (reduced units)
    constexpr double maxTime = 1e6;

    // Logging
    constexpr bool logInitialConditions = false;
    constexpr int progressLogEvery = 100;

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

    // XY-boundary: stop if |qx| or |qy| exceeds 10 Bohr radii (reduced units)
    inline const double xyBoundary = 10.0 * reducedBohr;

    // Detector: 1m distance, 1mm × 1mm square aperture centered on z-axis
    constexpr double detectorDistanceM = 1.0;      // meters
    constexpr double apertureHalfM = 50e-3;         // half-width in meters (100mm side)

} // namespace PhysicalData
