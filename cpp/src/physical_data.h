#pragma once

#include <cmath>
#include <array>
#include <string>
#include <cstdio>
#include <cstdlib>

namespace PhysicalData {

    // Integrator choice: Boost or DP853
    enum class Integrator { Boost, DP853 };
    constexpr Integrator integrator = Integrator::DP853;

    // Dual-row geometry: two rows of atoms separated by 1.42 Å in x
    // Each row at x = ±halfSeparation from channel center (x=0)
    constexpr double rowSeparationMeters = 1.42e-10;  // same as C-C bond length
    constexpr double halfSeparationMeters = rowSeparationMeters / 2.0;
    constexpr double halfSeparation = halfSeparationMeters / zitterRadius;  // ~367.8 reduced units
    constexpr int rowCount = 2;
    // Atom x-positions for the two rows (reduced units)
    inline const std::array<double, rowCount> atomX = { -halfSeparation, +halfSeparation };

    // Impact parameter range in meters: 0 to row wall minus one zitter radius
    constexpr double rangeMin = 0.0;
    constexpr double rangeMax = halfSeparationMeters - zitterRadius;  // 7.081e-11 m

    // Spin axis: set at runtime via CLI (default: +z)
    inline double spinTheta0 = 0.0;
    inline double spinPhi0 = 0.0;
    inline std::string spinLabel = "+z";

    inline bool spinRandom = false;  // when true, each electron gets a random spin axis
    inline bool noZitter = false;    // when true, charge locked to mass (classical point particle)

    inline void setSpinAxis(const std::string& axis) {
        if (axis == "+z" || axis == "z")       { spinTheta0 = 0.0;        spinPhi0 = 0.0;           spinLabel = "+z"; }
        else if (axis == "-z")                 { spinTheta0 = M_PI;       spinPhi0 = 0.0;           spinLabel = "-z"; }
        else if (axis == "+x" || axis == "x")  { spinTheta0 = M_PI/2.0;  spinPhi0 = 0.0;           spinLabel = "+x"; }
        else if (axis == "-x")                 { spinTheta0 = M_PI/2.0;  spinPhi0 = M_PI;          spinLabel = "-x"; }
        else if (axis == "+y" || axis == "y")  { spinTheta0 = M_PI/2.0;  spinPhi0 = M_PI/2.0;     spinLabel = "+y"; }
        else if (axis == "-y")                 { spinTheta0 = M_PI/2.0;  spinPhi0 = 3.0*M_PI/2.0;  spinLabel = "-y"; }
        else if (axis == "random" || axis == "rand") { spinRandom = true; spinLabel = "random"; }
        else {
            fprintf(stderr, "Unknown spin axis: '%s'. Use: +x,-x,+y,-y,+z,-z,random\n", axis.c_str());
            exit(1);
        }
    }

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
    constexpr int totalSimulations = 24;
    constexpr int plotsToShow = 10;

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

    // XY-boundary: stop if |qx| or |qy| exceeds 3 Bohr radii (reduced units)
    inline const double xyBoundary = 3.0 * reducedBohr;

} // namespace PhysicalData
