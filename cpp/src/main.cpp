#include "physical_data.h"
#include "electron.h"
#include "rivas_equations.h"
#include "dp853_integrator.h"

// Conditionally include integrator headers
#if __has_include("capd/capdlib.h")
#include "capd/capdlib.h"
#define HAVE_CAPD 1
#else
#define HAVE_CAPD 0
#endif

#include <boost/numeric/odeint.hpp>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <chrono>
#include <random>
#include <atomic>
#include <mutex>
#include <thread>
#include <cmath>
#include <ctime>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef HAVE_SFML
#include "PlotDots.h"
#include <X11/Xlib.h>
#endif

// ============================================================================
// CAPD vector field string — 30-atom carbon chain along z-axis
// Generated programmatically: each dv/dt component sums 30 screened Coulomb
// terms, one per atom at position ak. Atom positions passed as parameters.
// Sign: NEGATIVE dv/dt for attractive electron-nucleus Coulomb interaction
// ============================================================================
#if HAVE_CAPD
using namespace capd;

/**
 * Build the CAPD vector field string for N atoms along the z-axis.
 * Parameters: Z, alpha, rB, a0..a(N-1) where ak = z-position of atom k.
 *
 * For each atom k, the displacement vector is d = (r1, r2, r3-ak).
 * dv_i/dt = sum_k [ 2*Z*alpha * (d_i - v_i*(d.v)) / |d|^3 * sqrt(1-v^2) * exp(-|d|/rB) ]
 * du_i/dt = (q_i - r_i) * (1 - v.u) / |q - r|^2   (unchanged, potential-independent)
 */
static std::string buildRivasVectorField() {
    const int N = PhysicalData::atomCount;

    // Parameters: Z, alpha, rB, a0..a(N-1)
    std::string s = "par:Z,alpha,rB";
    for (int k = 0; k < N; k++) s += ",a" + std::to_string(k);
    s += ";var:q1,q2,q3,r1,r2,r3,v1,v2,v3,u1,u2,u3;fun:";

    // dq/dt = v, dr/dt = u
    s += "v1,v2,v3,u1,u2,u3,";

    // Helper lambdas for building per-atom expressions
    auto ak = [](int k) { return "a" + std::to_string(k); };
    auto dist2 = [&](int k) { return "r1^2+r2^2+(r3-" + ak(k) + ")^2"; };
    auto dist  = [&](int k) { return "((" + dist2(k) + ")^(0.5))"; };
    auto ddotv = [&](int k) { return "r1*v1+r2*v2+(r3-" + ak(k) + ")*v3"; };
    auto common = [&](int k) {
        return "((" + dist2(k) + ")^(-1.5))*((1.0-v1^2-v2^2-v3^2)^(0.5))*exp(-(" + dist(k) + ")/rB)";
    };

    // dv1/dt: x-component (r1 numerator unchanged — atoms are on z-axis)
    // NEGATIVE sign: attractive electron-nucleus Coulomb interaction
    for (int k = 0; k < N; k++) {
        if (k > 0) s += "+";
        s += "(-2)*Z*alpha*(r1-v1*(" + ddotv(k) + "))*" + common(k);
    }
    s += ",";

    // dv2/dt: y-component (r2 numerator unchanged)
    for (int k = 0; k < N; k++) {
        if (k > 0) s += "+";
        s += "(-2)*Z*alpha*(r2-v2*(" + ddotv(k) + "))*" + common(k);
    }
    s += ",";

    // dv3/dt: z-component (numerator has (r3-ak) shift)
    for (int k = 0; k < N; k++) {
        if (k > 0) s += "+";
        s += "(-2)*Z*alpha*((r3-" + ak(k) + ")-v3*(" + ddotv(k) + "))*" + common(k);
    }
    s += ",";

    // du/dt: zitter constraint (unchanged, no potential dependence)
    s += "(q1-r1)*(1-v1*u1-v2*u2-v3*u3)/((q1-r1)^2+(q2-r2)^2+(q3-r3)^2),";
    s += "(q2-r2)*(1-v1*u1-v2*u2-v3*u3)/((q1-r1)^2+(q2-r2)^2+(q3-r3)^2),";
    s += "(q3-r3)*(1-v1*u1-v2*u2-v3*u3)/((q1-r1)^2+(q2-r2)^2+(q3-r3)^2);";

    return s;
}
#endif

struct SimulationResult {
    Electron electron;
    long elapsedMs;
    bool detected = false;
};

// ============================================================================
// CAPD Taylor integrator path
// ============================================================================
#if HAVE_CAPD
SimulationResult runCapd(double rangeMin, double rangeMax, std::mt19937& rng, bool recordCamera = false) {

    Electron electron(PhysicalData::startEnergy, rangeMin, rangeMax, rng);
    electron.recordCamera = recordCamera;
    auto startTime = std::chrono::steady_clock::now();

    // Build and parse CAPD vector field string for N-atom chain
    // CAPD string parsing is NOT thread-safe — protect with critical section
    DMap* pVectorField;
    DOdeSolver* pSolver;
    DTimeMap* pTimeMap;
    #pragma omp critical(capd_init)
    {
        std::string vfString = buildRivasVectorField();
        pVectorField = new DMap(vfString);
        pVectorField->setParameter("Z", PhysicalData::carbonProtons);
        pVectorField->setParameter("alpha", PhysicalData::alpha);
        pVectorField->setParameter("rB", PhysicalData::reducedBohr);
        for (int k = 0; k < PhysicalData::atomCount; k++) {
            pVectorField->setParameter("a" + std::to_string(k), PhysicalData::atomZ[k]);
        }
        pSolver = new DOdeSolver(*pVectorField, PhysicalData::capdOrder);
        pTimeMap = new DTimeMap(*pSolver);
    }
    DTimeMap& timeMap = *pTimeMap;

    // Load initial state into CAPD DVector
    DVector state(12);
    for (int i = 0; i < 12; i++) state[i] = electron.currentState[i];

    electron.storePoint();

    // Helper: distance from a point to the nearest atom
    auto nearestAtomDist = [](double px, double py, double pz) {
        double xy2 = px*px + py*py;
        double minDist = 1e30;
        for (int k = 0; k < PhysicalData::atomCount; k++) {
            double dz = pz - PhysicalData::atomZ[k];
            double d = std::sqrt(xy2 + dz*dz);
            if (d < minDist) minDist = d;
        }
        return minDist;
    };

    bool stopped = false;
    double t = 0.0;
    int capdCameraCount = 0;
    // CAPD takes ~4200 steps (vs Boost ~13.7M), so use very low decimation
    static constexpr int CAPD_CAMERA_DECIMATION = 2;

    try {
        while (t < PhysicalData::maxTime && !stopped) {

            // Adaptive step based on distance to nearest atom (charge center)
            // Tighter steps near nuclei to handle attractive 1/r^3 singularity
            double distCharge = nearestAtomDist(state[3], state[4], state[5]);
            double divisor;
            if (distCharge < 1.0)        divisor = 200.0;   // very close approach
            else if (distCharge < 5.0)   divisor = 100.0;   // close approach
            else if (distCharge < 20.0)  divisor = 50.0;    // near atom
            else if (distCharge < 100.0) divisor = 10.0;    // moderate distance
            else                         divisor = 2.0;     // far from all atoms
            double dt = distCharge / divisor;
            if (dt > PhysicalData::maxStep) dt = PhysicalData::maxStep;
            if (dt < PhysicalData::capdMinStep) dt = PhysicalData::capdMinStep;

            timeMap(dt, state);
            t += dt;

            // Copy back to electron for history/diagnostics
            State s;
            for (int i = 0; i < 12; i++) s[i] = state[i];
            electron.loadState(s);

            // CAPD camera: low decimation since CAPD takes ~4200 steps (vs Boost ~13.7M)
            // Capture while electron z-position is within the atom chain range
            if (recordCamera && state[2] >= Electron::CAMERA_Z_MIN && state[2] <= Electron::CAMERA_Z_MAX) {
                if (++capdCameraCount >= CAPD_CAMERA_DECIMATION) {
                    capdCameraCount = 0;
                    electron.stateCamera.push_back(s);
                }
            }

            electron.storePoint();

            // Check forward detection: qz beyond chain
            if (state[2] > PhysicalData::detectionDistance) stopped = true;

            // Check backward detection: qz behind chain and heading away (vz < 0)
            if (state[2] < -PhysicalData::detectionDistance && state[8] < 0) stopped = true;

            // Check xy-boundary: |qx| or |qy| > 10 Bohr radii
            if (std::abs(state[0]) > PhysicalData::xyBoundary || std::abs(state[1]) > PhysicalData::xyBoundary) stopped = true;

            // Check superluminal: v^2 >= 0.9999
            double v2 = state[6]*state[6] + state[7]*state[7] + state[8]*state[8];
            if (v2 > 0.9999) { stopped = true; }
        }
    } catch (std::exception& e) {
        std::cerr << "CAPD exception: " << e.what() << "\n";
    } catch (...) {
        std::cerr << "CAPD unknown exception\n";
    }

    // Final state
    State finalState;
    for (int i = 0; i < 12; i++) finalState[i] = state[i];
    electron.loadState(finalState);

    delete pTimeMap;
    delete pSolver;
    delete pVectorField;

    auto endTime = std::chrono::steady_clock::now();
    long ms = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    return {std::move(electron), ms};
}
#endif

// ============================================================================
// Boost.Odeint Dormand-Prince 5(4) integrator path
// ============================================================================
SimulationResult runBoost(double rangeMin, double rangeMax, std::mt19937& rng, bool recordCamera = false) {

    using namespace boost::numeric::odeint;
    typedef runge_kutta_dopri5<State> dopri5_type;
    typedef controlled_runge_kutta<dopri5_type> controlled_type;

    Electron electron(PhysicalData::startEnergy, rangeMin, rangeMax, rng);
    electron.recordCamera = recordCamera;
    auto startTime = std::chrono::steady_clock::now();

    RivasEquations equations;
    controlled_type stepper = make_controlled(PhysicalData::boostAbsTol,
                                              PhysicalData::boostRelTol,
                                              dopri5_type());

    State state = electron.currentState;
    electron.storePoint();

    bool stopped = false;
    double t = 0.0;
    double dt = 0.01; // initial step — adaptive stepper will adjust

    try {
        while (t < PhysicalData::maxTime && !stopped) {

            // Boost adaptive step
            controlled_step_result stepResult;
            do {
                stepResult = stepper.try_step(equations, state, t, dt);
            } while (stepResult == fail);
            // On success, t and dt are updated by try_step

            // Copy to electron
            electron.loadState(state);
            electron.storePoint();

            // Check forward detection: qz beyond chain
            if (state[QZ] > PhysicalData::detectionDistance) stopped = true;

            // Check backward detection: qz behind chain and heading away (vz < 0)
            if (state[QZ] < -PhysicalData::detectionDistance && state[VZ] < 0) stopped = true;

            // Check xy-boundary: |qx| or |qy| > 10 Bohr radii
            if (std::abs(state[QX]) > PhysicalData::xyBoundary || std::abs(state[QY]) > PhysicalData::xyBoundary) stopped = true;

            // Check superluminal: v^2 >= 0.9999
            double v2 = state[VX]*state[VX] + state[VY]*state[VY] + state[VZ]*state[VZ];
            if (v2 > 0.9999) { stopped = true; }
        }
    } catch (std::exception& e) {
        std::cerr << "Boost exception: " << e.what() << "\n";
    } catch (...) {
        std::cerr << "Boost unknown exception\n";
    }

    electron.loadState(state);

    auto endTime = std::chrono::steady_clock::now();
    long ms = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    return {std::move(electron), ms};
}

// ============================================================================
// Self-contained Dormand-Prince 8(5,3) integrator path
// Matches Apache commons-math3 DormandPrince853Integrator exactly
// ============================================================================
SimulationResult runDP853(double rangeMin, double rangeMax, std::mt19937& rng, bool recordCamera = false) {

    Electron electron(PhysicalData::startEnergy, rangeMin, rangeMax, rng);
    electron.recordCamera = recordCamera;
    auto startTime = std::chrono::steady_clock::now();

    RivasEquations equations;
    DP853Integrator<12> integrator(
        PhysicalData::dp853MinStep, PhysicalData::dp853MaxStep,
        PhysicalData::dp853AbsTol, PhysicalData::dp853RelTol);

    State state = electron.currentState;
    electron.storePoint();

    // Wrap RivasEquations for DP853's (t, y*, dydt*) calling convention
    auto rhs = [&equations](double t, const double* y, double* dydt) {
        State yState, dydtState;
        std::memcpy(yState.data(), y, 12 * sizeof(double));
        equations(yState, dydtState, t);
        std::memcpy(dydt, dydtState.data(), 12 * sizeof(double));
    };

    bool stopped = false;
    auto callback = [&](double t, const double* y) -> bool {
        State s;
        std::memcpy(s.data(), y, 12 * sizeof(double));
        electron.loadState(s);
        electron.storePoint();

        // Forward detection
        if (s[QZ] > PhysicalData::detectionDistance) return false;
        // Backward detection
        if (s[QZ] < -PhysicalData::detectionDistance && s[VZ] < 0) return false;
        // XY-boundary: |qx| or |qy| > 10 Bohr radii
        if (std::abs(s[QX]) > PhysicalData::xyBoundary || std::abs(s[QY]) > PhysicalData::xyBoundary) return false;
        // Superluminal check
        double v2 = s[VX]*s[VX] + s[VY]*s[VY] + s[VZ]*s[VZ];
        if (v2 > 0.9999) { return false; }

        return true;
    };

    try {
        integrator.integrate(rhs, 0.0, state.data(), PhysicalData::maxTime, callback);
    } catch (std::exception& e) {
        std::cerr << "DP853 exception: " << e.what() << "\n";
    } catch (...) {
        std::cerr << "DP853 unknown exception\n";
    }

    // Final state (already loaded via callback, but ensure consistency)
    electron.loadState(state);
    electron.internalCount = integrator.nSteps;

    auto endTime = std::chrono::steady_clock::now();
    long ms = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    return {std::move(electron), ms};
}

// ============================================================================
// Dispatcher — compile-time integrator selection
// ============================================================================
SimulationResult runSingleSimulation(double rangeMin, double rangeMax, std::mt19937& rng, bool recordCamera = false) {
    if constexpr (PhysicalData::integrator == PhysicalData::Integrator::CAPD) {
#if HAVE_CAPD
        return runCapd(rangeMin, rangeMax, rng, recordCamera);
#else
        static_assert(false, "CAPD selected but capd/capdlib.h not found. Install CAPD or switch to Boost/DP853.");
#endif
    } else if constexpr (PhysicalData::integrator == PhysicalData::Integrator::DP853) {
        return runDP853(rangeMin, rangeMax, rng, recordCamera);
    } else {
        return runBoost(rangeMin, rangeMax, rng, recordCamera);
    }
}

// ============================================================================
// Main
// ============================================================================
int main() {

#ifdef HAVE_SFML
    XInitThreads();  // Required for multi-threaded X11/SFML windows
#endif

    int totalSimulations = PhysicalData::totalSimulations;
    int plotsToShow = PhysicalData::plotsToShow;

    int cores = 1;
#ifdef _OPENMP
    cores = omp_get_max_threads();
#endif

    std::cout << "PARAMS | rangeMin: " << PhysicalData::rangeMin
              << " | rangeMax: " << PhysicalData::rangeMax
              << " | startEnergy: " << PhysicalData::startEnergy
              << " | spin: " << PhysicalData::spin
              << " | carbonProtons(Z): " << PhysicalData::carbonProtons
              << " | atoms: " << PhysicalData::atomCount
              << " | spacing: " << PhysicalData::atomSpacing << " (reduced)\n";

    if constexpr (PhysicalData::integrator == PhysicalData::Integrator::CAPD) {
        std::cout << "Integrator: CAPD Taylor (order " << PhysicalData::capdOrder << ")"
                  << " | stepDivisor: " << PhysicalData::capdStepDivisor << "\n";
    } else if constexpr (PhysicalData::integrator == PhysicalData::Integrator::DP853) {
        std::cout << "Integrator: DP853 (self-contained)"
                  << " | absTol: " << PhysicalData::dp853AbsTol
                  << " | relTol: " << PhysicalData::dp853RelTol
                  << " | minStep: " << PhysicalData::dp853MinStep
                  << " | maxStep: " << PhysicalData::dp853MaxStep << "\n";
    } else {
        std::cout << "Integrator: Boost.Odeint DormandPrince5(4)"
                  << " | absTol: " << PhysicalData::boostAbsTol
                  << " | relTol: " << PhysicalData::boostRelTol << "\n";
    }

    std::cout << "Running " << totalSimulations << " simulations on " << cores << " cores.\n";
    std::cout << "Detector: " << PhysicalData::detectorDistanceM << "m distance, "
              << (PhysicalData::apertureHalfM * 2000.0) << "mm x "
              << (PhysicalData::apertureHalfM * 2000.0) << "mm aperture\n";

    auto totalStart = std::chrono::steady_clock::now();

    // Collect results
    std::vector<SimulationResult> results(totalSimulations);
    std::atomic<int> completedCount{0};
    std::atomic<int> detectedCount{0};
    std::mutex printMutex;

    #pragma omp parallel
    {
        // Each thread gets its own RNG seeded differently
        unsigned int seed = std::random_device{}();
        #ifdef _OPENMP
        seed += omp_get_thread_num();
        #endif
        std::mt19937 rng(seed);

        #pragma omp for schedule(dynamic)
        for (int i = 0; i < totalSimulations; i++) {
            results[i] = runSingleSimulation(PhysicalData::rangeMin, PhysicalData::rangeMax, rng, true);

            int count = ++completedCount;

            // Check detector hit: forward electron, project to detector plane
            auto& e = results[i].electron;
            const State& s = e.currentState;
            if (s[VZ] > 0) {
                double xAtDet = (s[VX] / s[VZ]) * PhysicalData::detectorDistanceM;
                double yAtDet = (s[VY] / s[VZ]) * PhysicalData::detectorDistanceM;
                if (std::abs(xAtDet) < PhysicalData::apertureHalfM &&
                    std::abs(yAtDet) < PhysicalData::apertureHalfM) {
                    results[i].detected = true;
                    int det = ++detectedCount;
                    std::lock_guard<std::mutex> lock(printMutex);
                    std::cout << "DETECTED #" << det << " (electron " << i << ")"
                              << " | Steps: " << e.internalCount
                              << e.getEXIT()
                              << " | xDet: " << std::scientific << std::setprecision(3) << xAtDet * 1e3 << "mm"
                              << " | yDet: " << yAtDet * 1e3 << "mm"
                              << " | Time: " << results[i].elapsedMs << "ms\n";
                }
            }

            if (count % 48 == 0 || count == totalSimulations) {
                std::lock_guard<std::mutex> lock(printMutex);
                std::cout << "Progress: " << count << "/" << totalSimulations
                          << " | Detected: " << detectedCount.load() << "\n";
            }
        }
    }

    auto totalEnd = std::chrono::steady_clock::now();
    long totalMs = std::chrono::duration_cast<std::chrono::milliseconds>(totalEnd - totalStart).count();

    std::cout << "\nTOTAL TIME FOR " << totalSimulations << " SIMULATIONS: "
              << totalMs << "ms (" << cores << " cores)"
              << " | DETECTED: " << detectedCount.load() << "/" << totalSimulations << "\n";

    // ================================================================
    // Write full-precision results file for DETECTED electrons only
    // ================================================================
    {
        // Timestamp
        auto now = std::chrono::system_clock::now();
        std::time_t nowTime = std::chrono::system_clock::to_time_t(now);
        char timeBuf[64], dateBuf[64], timeFmt[64];
        std::strftime(timeBuf, sizeof(timeBuf), "%Y-%m-%d %H:%M:%S", std::localtime(&nowTime));
        std::strftime(dateBuf, sizeof(dateBuf), "%Y-%m-%d", std::localtime(&nowTime));
        std::strftime(timeFmt, sizeof(timeFmt), "%H%M%S", std::localtime(&nowTime));

        std::string integratorName, integratorTag;
        if constexpr (PhysicalData::integrator == PhysicalData::Integrator::CAPD) {
            integratorName = "CAPD Taylor (order " + std::to_string(PhysicalData::capdOrder) + ")";
            integratorTag = "capd";
        } else if constexpr (PhysicalData::integrator == PhysicalData::Integrator::DP853) {
            integratorName = "DP853 (self-contained)";
            integratorTag = "dp853";
        } else {
            integratorName = "Boost.Odeint DormandPrince5(4)";
            integratorTag = "boost";
        }

        // results/<date>_<time>_cpp-<integrator>_<iterations>.dat
        // Write to project root /results/ via relative path from build dir
        std::string resultsDir = "/mnt/c/Users/marcf/IdeaProjects/ELektron2/results/";
        std::string resultsFile = std::string(dateBuf) + "_" + timeFmt
            + "_cpp-" + integratorTag + "_" + std::to_string(totalSimulations) + ".dat";
        std::string resultsPath = resultsDir + resultsFile;

        std::ofstream out(resultsPath);
        if (!out.is_open()) {
            // Fallback to current directory
            resultsPath = resultsFile;
            out.open(resultsPath);
        }
        out << std::setprecision(std::numeric_limits<double>::max_digits10);

        // Context header
        out << "# ELektron2 C++ Simulation Results\n";
        out << "# Date: " << timeBuf << "\n";
        out << "# Integrator: " << integratorName << "\n";
        if constexpr (PhysicalData::integrator == PhysicalData::Integrator::CAPD) {
            out << "# CAPD order: " << PhysicalData::capdOrder
                << "  stepDivisor: " << PhysicalData::capdStepDivisor
                << "  minStep: " << PhysicalData::capdMinStep
                << "  maxStep: " << PhysicalData::maxStep << "\n";
        } else if constexpr (PhysicalData::integrator == PhysicalData::Integrator::DP853) {
            out << "# DP853 absTol: " << PhysicalData::dp853AbsTol
                << "  relTol: " << PhysicalData::dp853RelTol
                << "  minStep: " << PhysicalData::dp853MinStep
                << "  maxStep: " << PhysicalData::dp853MaxStep << "\n";
        } else {
            out << "# Boost absTol: " << PhysicalData::boostAbsTol
                << "  relTol: " << PhysicalData::boostRelTol << "\n";
        }
        out << "# Cores: " << cores << "\n";
        out << "# Total time: " << totalMs << " ms\n";
        out << "# Total simulations: " << totalSimulations << "\n";
        out << "# Detected: " << detectedCount.load() << "\n";
        out << "# Detector: " << PhysicalData::detectorDistanceM << " m, "
            << (PhysicalData::apertureHalfM * 2000.0) << " mm x "
            << (PhysicalData::apertureHalfM * 2000.0) << " mm aperture\n";
        out << "# startEnergy: " << PhysicalData::startEnergy << " eV\n";
        out << "# startPos: " << PhysicalData::startPos << " (reduced)\n";
        out << "# detectionDistance: " << PhysicalData::detectionDistance << " (reduced)\n";
        out << "# rangeMin: " << PhysicalData::rangeMin << " m\n";
        out << "# rangeMax: " << PhysicalData::rangeMax << " m\n";
        out << "# spin: " << PhysicalData::spin << "\n";
        out << "# Z: " << PhysicalData::carbonProtons << "\n";
        out << "# alpha: " << PhysicalData::alpha << "\n";
        out << "# reducedBohr: " << PhysicalData::reducedBohr << "\n";
        out << "# zitterRadius: " << PhysicalData::zitterRadius << " m\n";
        out << "# atomCount: " << PhysicalData::atomCount << "\n";
        out << "# atomSpacing: " << PhysicalData::atomSpacing << " (reduced) = "
            << PhysicalData::atomSpacingMeters << " m\n";
        out << "# chainHalfLength: " << PhysicalData::chainHalfLength << " (reduced)\n";
        out << "# maxTime: " << PhysicalData::maxTime << " (reduced)\n";
        out << "#\n";
        out << "# Columns:\n";
        out << "# idx qx qy qz rx ry rz vx vy vz ux uy uz"
            << " energyIn_eV energyOut_eV angle_deg steps"
            << " elapsedMs dxZERO_reduced psi0"
            << " xDet_mm yDet_mm\n";
        out << "#\n";

        int written = 0;
        for (int i = 0; i < totalSimulations; i++) {
            if (!results[i].detected) continue;
            auto& e = results[i].electron;
            const State& s = e.currentState;
            double xAtDet = (s[VX] / s[VZ]) * PhysicalData::detectorDistanceM;
            double yAtDet = (s[VY] / s[VZ]) * PhysicalData::detectorDistanceM;

            out << written
                << " " << s[QX] << " " << s[QY] << " " << s[QZ]
                << " " << s[RX] << " " << s[RY] << " " << s[RZ]
                << " " << s[VX] << " " << s[VY] << " " << s[VZ]
                << " " << s[UX] << " " << s[UY] << " " << s[UZ]
                << " " << e.initialKineticEnergy
                << " " << e.getKineticEnergy()
                << " " << e.getAngle()
                << " " << e.internalCount
                << " " << results[i].elapsedMs
                << " " << e.dxZERO
                << " " << e.psi0
                << " " << xAtDet * 1e3
                << " " << yAtDet * 1e3
                << "\n";
            written++;
        }

        out.close();
        std::cout << "Wrote " << written << " detected electrons (of " << totalSimulations
                  << " total) to " << resultsPath << "\n";
    }

    // ================================================================
    // Show PlotDots visualization for DETECTED electrons only
    // Each window runs in its own thread — all visible simultaneously
    // ================================================================
#ifdef HAVE_SFML
    PlotDots::closeAll.store(false);
    std::vector<std::thread> plotThreads;
    for (int i = 0; i < totalSimulations; i++) {
        if (results[i].detected && results[i].electron.stateCamera.size() >= 2) {
            std::cout << "Launching PlotDots for detected electron " << i
                      << " (" << results[i].electron.stateCamera.size() << " camera points)\n";
            plotThreads.emplace_back([&results, i]() {
                PlotDots::show(results[i].electron);
            });
        }
    }
    for (auto& t : plotThreads) t.join();
#endif

    return 0;
}
