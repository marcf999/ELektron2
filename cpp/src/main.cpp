#include "physical_data.h"
#include "electron.h"
#include "rivas_equations.h"
#include "dp853_integrator.h"

#include <boost/numeric/odeint.hpp>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <algorithm>
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

struct SimulationResult {
    Electron electron;
    long elapsedMs;
};

// ============================================================================
// Boost.Odeint Dormand-Prince 5(4) integrator path
// ============================================================================
SimulationResult runBoost(double energyEV, double rangeMin, double rangeMax, std::mt19937& rng, bool recordCamera = false) {

    using namespace boost::numeric::odeint;
    typedef runge_kutta_dopri5<State> dopri5_type;
    typedef controlled_runge_kutta<dopri5_type> controlled_type;

    Electron electron(energyEV, rangeMin, rangeMax, rng);
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
SimulationResult runDP853(double energyEV, double rangeMin, double rangeMax, std::mt19937& rng, bool recordCamera = false) {

    Electron electron(energyEV, rangeMin, rangeMax, rng);
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
SimulationResult runSingleSimulation(double energyEV, double rangeMin, double rangeMax, std::mt19937& rng, bool recordCamera = false) {
    if constexpr (PhysicalData::integrator == PhysicalData::Integrator::DP853) {
        return runDP853(energyEV, rangeMin, rangeMax, rng, recordCamera);
    } else {
        return runBoost(energyEV, rangeMin, rangeMax, rng, recordCamera);
    }
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {

#ifdef HAVE_SFML
    XInitThreads();  // Required for multi-threaded X11/SFML windows
#endif

    int totalSimulations = PhysicalData::totalSimulations;
    int plotsToShow = PhysicalData::plotsToShow;

    // Parse args: elektron2 [numElectrons] [energyEV] [spinAxis]
    //   spinAxis: +x, -x, +y, -y, +z, -z (default: +z)
    if (argc > 1) totalSimulations = std::atoi(argv[1]);
    if (totalSimulations < 1) totalSimulations = PhysicalData::totalSimulations;
    double energyEV = PhysicalData::startEnergy;
    if (argc > 2) energyEV = std::atof(argv[2]);
    if (energyEV <= 0) energyEV = PhysicalData::startEnergy;
    if (argc > 3) PhysicalData::setSpinAxis(argv[3]);

    int cores = 1;
#ifdef _OPENMP
    cores = omp_get_max_threads();
#endif

    std::cout << "PARAMS | rangeMin: " << PhysicalData::rangeMin
              << " | rangeMax: " << PhysicalData::rangeMax
              << " | startEnergy: " << energyEV
              << " | spin: " << PhysicalData::spinLabel
              << " | carbonProtons(Z): " << PhysicalData::carbonProtons
              << " | atoms: " << PhysicalData::atomCount
              << " | spacing: " << PhysicalData::atomSpacing << " (reduced)\n";

    if constexpr (PhysicalData::integrator == PhysicalData::Integrator::DP853) {
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

    auto totalStart = std::chrono::steady_clock::now();

    // Collect results
    std::vector<SimulationResult> results(totalSimulations);
    std::atomic<int> completedCount{0};
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
            results[i] = runSingleSimulation(energyEV, PhysicalData::rangeMin, PhysicalData::rangeMax, rng, true);

            int count = ++completedCount;

            if (count % 48 == 0 || count == totalSimulations) {
                auto now = std::chrono::steady_clock::now();
                long elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(now - totalStart).count();
                std::lock_guard<std::mutex> lock(printMutex);
                std::cout << "Progress: " << count << "/" << totalSimulations
                          << " | Elapsed: " << std::fixed << std::setprecision(1) << elapsedMs / 1000.0 << "s\n";
                std::cout << std::defaultfloat;
            }
        }
    }

    auto totalEnd = std::chrono::steady_clock::now();
    long totalMs = std::chrono::duration_cast<std::chrono::milliseconds>(totalEnd - totalStart).count();

    std::cout << "\nTOTAL TIME FOR " << totalSimulations << " SIMULATIONS: "
              << totalMs << "ms (" << cores << " cores)"
              << " | Energy: " << energyEV << " eV\n";
    std::cout << std::defaultfloat;

    // ================================================================
    // Write full-precision results file for forward-exit electrons only
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
        if constexpr (PhysicalData::integrator == PhysicalData::Integrator::DP853) {
            integratorName = "DP853 (self-contained)";
            integratorTag = "dp853";
        } else {
            integratorName = "Boost.Odeint DormandPrince5(4)";
            integratorTag = "boost";
        }

        // results/<date>_<time>_cpp-<integrator>_<energy>_<iterations>.dat
        // Write to project root /results/ via relative path from build dir
        char energyStr[32];
        std::snprintf(energyStr, sizeof(energyStr), "%.0feV", energyEV);
        std::string resultsDir = "/mnt/c/Users/marcf/IdeaProjects/ELektron2/results/";
        std::string resultsFile = std::string(dateBuf) + "_" + timeFmt
            + "_cpp-" + integratorTag + "_" + energyStr + "_" + std::to_string(totalSimulations) + ".dat";
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
        if constexpr (PhysicalData::integrator == PhysicalData::Integrator::DP853) {
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
        out << "# startEnergy: " << energyEV << " eV\n";
        out << "# startPos: " << PhysicalData::startPos << " (reduced)\n";
        out << "# detectionDistance: " << PhysicalData::detectionDistance << " (reduced)\n";
        out << "# rangeMin: " << PhysicalData::rangeMin << " m\n";
        out << "# rangeMax: " << PhysicalData::rangeMax << " m\n";
        out << "# spinOrientation: " << PhysicalData::spinLabel << "\n";
        out << "# theta0: " << PhysicalData::spinTheta0 << " rad  (polar angle of spin axis)\n";
        out << "# phi0: " << PhysicalData::spinPhi0 << " rad  (azimuthal angle of spin axis)\n";
        out << "# psi0: random [0, 2pi)  (zitter phase, per-electron)\n";
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
            << " elapsedMs dxZERO_reduced psi0\n";
        out << "#\n";

        int written = 0;
        for (int i = 0; i < totalSimulations; i++) {
            auto& e = results[i].electron;
            const State& s = e.currentState;

            if (s[QZ] < PhysicalData::detectionDistance - 1.0) continue;

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
                << "\n";
            written++;
        }

        out.close();
        std::cout << "Wrote " << written << " electrons (of " << totalSimulations
                  << " total) to " << resultsPath << "\n";
    }

    // ================================================================
    // Show PlotDots visualization for first N forward-exit electrons
    // Each window runs in its own thread — all visible simultaneously
    // ================================================================
#ifdef HAVE_SFML
    PlotDots::closeAll.store(false);
    std::vector<std::thread> plotThreads;
    int plotted = 0;
    for (int i = 0; i < totalSimulations && plotted < PhysicalData::plotsToShow; i++) {
        if (results[i].electron.stateCamera.size() >= 2) {
            plotted++;
            std::cout << "Launching PlotDots for electron " << i
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
