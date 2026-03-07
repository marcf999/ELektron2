#include "physical_data.h"
#include "electron.h"
#include "rivas_equations.h"

#include <boost/numeric/odeint.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <random>
#include <atomic>
#include <mutex>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace odeint = boost::numeric::odeint;

// Custom observer that stores points and checks termination events
struct SimObserver {
    Electron& electron;
    bool stopped = false;
    std::string stopReason;

    SimObserver(Electron& e) : electron(e) {}

    void operator()(const State& y, double /*t*/) {
        electron.loadState(y);
        electron.storePoint();
        if (PhysicalData::debug) electron.debugUpdate();

        // Check forward detection: qz > +1000
        if (y[QZ] > PhysicalData::detectionDistance) {
            stopped = true;
            stopReason = "forward";
        }

        // Check backward detection: qz < -1000 and heading away (vz < 0)
        if (y[QZ] < -PhysicalData::detectionDistance && y[VZ] < 0) {
            stopped = true;
            stopReason = "backward";
        }

        // Check superluminal: v^2 >= 0.9999
        double v2 = y[VX]*y[VX] + y[VY]*y[VY] + y[VZ]*y[VZ];
        if (v2 > 0.9999) {
            electron.isNaN = true;
            stopped = true;
            stopReason = "superluminal";
        }
    }
};

struct SimulationResult {
    Electron electron;
    long elapsedMs;
};

SimulationResult runSingleSimulation(double rangeMin, double rangeMax, std::mt19937& rng) {

    Electron electron(PhysicalData::startEnergy, rangeMin, rangeMax, rng);
    RivasEquations equations;

    auto startTime = std::chrono::steady_clock::now();

    // Boost.Odeint Dormand-Prince 8(5,3) — equivalent to DormandPrince853
    using stepper_type = odeint::runge_kutta_dopri5<State>;
    auto stepper = odeint::make_controlled<stepper_type>(PhysicalData::absTol, PhysicalData::relTol);

    State state = electron.currentState;
    double t = 0.0;
    double dt = 1.0; // initial step guess

    electron.storePoint();

    SimObserver observer(electron);

    try {
        while (t < PhysicalData::maxTime && !observer.stopped) {
            // Clamp step size
            if (dt < PhysicalData::minStep) dt = PhysicalData::minStep;
            if (dt > PhysicalData::maxStep) dt = PhysicalData::maxStep;

            auto result = stepper.try_step(equations, state, t, dt);

            if (result == odeint::controlled_step_result::success) {
                observer(state, t);
            }
            // If fail, stepper reduces dt automatically and we retry
        }
    } catch (...) {
        electron.isNaN = true;
    }

    electron.loadState(state);

    auto endTime = std::chrono::steady_clock::now();
    long ms = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

    return {std::move(electron), ms};
}

int main() {

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
              << " | carbonProtons(Z): " << PhysicalData::carbonProtons << "\n";
    std::cout << "Integrator: Boost.Odeint Dormand-Prince 5(4)"
              << " | relTol: " << PhysicalData::relTol
              << " | absTol: " << PhysicalData::absTol << "\n";
    std::cout << "Running " << totalSimulations << " simulations on " << cores << " cores.\n";

    auto totalStart = std::chrono::steady_clock::now();

    // Collect results
    std::vector<SimulationResult> results(totalSimulations);
    std::atomic<int> completedCount{0};
    std::mutex printMutex;

    int isNaN_total = 0, isRenorm_total = 0, isNeg_total = 0, isPos_total = 0;
    int is120L_total = 0, is120R_total = 0;

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
            results[i] = runSingleSimulation(PhysicalData::rangeMin, PhysicalData::rangeMax, rng);

            int count = ++completedCount;

            if (count % PhysicalData::progressLogEvery == 0 || count == totalSimulations) {
                auto& e = results[i].electron;
                std::lock_guard<std::mutex> lock(printMutex);
                std::cout << "RUNS FINISHED: " << count
                          << " | Steps: " << e.internalCount
                          << e.getEXIT()
                          << e.getConstraints()
                          << " | Time: " << results[i].elapsedMs << "ms\n";
            }
        }
    }

    // Tally results
    for (int i = 0; i < totalSimulations; i++) {
        auto& e = results[i].electron;
        if (e.isNaN) { isNaN_total++; continue; }
        if (e.isPos()) isPos_total++;
        if (e.isNeg()) isNeg_total++;
        if (e.is120R()) is120R_total++;
        if (e.is120L()) is120L_total++;
        if (e.isRenorm) isRenorm_total++;
    }

    auto totalEnd = std::chrono::steady_clock::now();
    long totalMs = std::chrono::duration_cast<std::chrono::milliseconds>(totalEnd - totalStart).count();

    std::cout << "\n=== SUMMARY ===\n"
              << "isNaN: " << isNaN_total
              << " | isPos: " << isPos_total
              << " | isNeg: " << isNeg_total
              << " | is120L: " << is120L_total
              << " | is120R: " << is120R_total
              << " | isRenorm: " << isRenorm_total << "\n";
    std::cout << "TOTAL TIME FOR " << totalSimulations << " SIMULATIONS: "
              << totalMs << "ms (" << cores << " cores)\n";

    // Write trajectory data for the first plotsToShow electrons
    int toWrite = std::min(plotsToShow, totalSimulations);
    for (int i = 0; i < toWrite; i++) {
        std::string filename = "trajectory_" + std::to_string(i) + ".dat";
        FILE* f = fopen(filename.c_str(), "w");
        if (!f) continue;
        fprintf(f, "# qx qy qz rx ry rz\n");
        for (const auto& s : results[i].electron.stateCamera) {
            fprintf(f, "%.10e %.10e %.10e %.10e %.10e %.10e\n",
                    s[QX], s[QY], s[QZ], s[RX], s[RY], s[RZ]);
        }
        fclose(f);
        std::cout << "Wrote " << results[i].electron.stateCamera.size()
                  << " points to " << filename << "\n";
    }

    std::cout << "\nTo plot: gnuplot -e \"plot 'trajectory_0.dat' using 1:3 with dots\"\n";

    return 0;
}
