#include "physical_data.h"
#include "electron.h"
#include "rivas_equations.h"

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
#include <vector>
#include <chrono>
#include <random>
#include <atomic>
#include <mutex>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

// ============================================================================
// CAPD vector field string — same as original ELektron
// Note: CAPD sign convention has POSITIVE dv/dt (see lab notes for discussion)
// ============================================================================
#if HAVE_CAPD
using namespace capd;

static const std::string RIVAS_VECTOR_FIELD =
    "par:Z,alpha,rB;var:q1,q2,q3,r1,r2,r3,v1,v2,v3,u1,u2,u3;fun:"
    "v1,v2,v3,"
    "u1,u2,u3,"
    "2*Z*alpha*(r1-v1*(r1*v1+r2*v2+r3*v3))*((r1^2+r2^2+r3^2)^(-1.5))*((1.0-v1^2-v2^2-v3^2)^(0.5))*exp(-((r1^2+r2^2+r3^2)^(0.5))/rB),"
    "2*Z*alpha*(r2-v2*(r1*v1+r2*v2+r3*v3))*((r1^2+r2^2+r3^2)^(-1.5))*((1.0-v1^2-v2^2-v3^2)^(0.5))*exp(-((r1^2+r2^2+r3^2)^(0.5))/rB),"
    "2*Z*alpha*(r3-v3*(r1*v1+r2*v2+r3*v3))*((r1^2+r2^2+r3^2)^(-1.5))*((1.0-v1^2-v2^2-v3^2)^(0.5))*exp(-((r1^2+r2^2+r3^2)^(0.5))/rB),"
    "(q1-r1)*(1-v1*u1-v2*u2-v3*u3)/((q1-r1)^2+(q2-r2)^2+(q3-r3)^2),"
    "(q2-r2)*(1-v1*u1-v2*u2-v3*u3)/((q1-r1)^2+(q2-r2)^2+(q3-r3)^2),"
    "(q3-r3)*(1-v1*u1-v2*u2-v3*u3)/((q1-r1)^2+(q2-r2)^2+(q3-r3)^2);";
#endif

struct SimulationResult {
    Electron electron;
    long elapsedMs;
};

// ============================================================================
// CAPD Taylor integrator path
// ============================================================================
#if HAVE_CAPD
SimulationResult runCapd(double rangeMin, double rangeMax, std::mt19937& rng) {

    Electron electron(PhysicalData::startEnergy, rangeMin, rangeMax, rng);
    auto startTime = std::chrono::steady_clock::now();

    // Create CAPD integrator: DMap -> DOdeSolver -> DTimeMap
    // CAPD string parsing is NOT thread-safe — protect with critical section
    DMap* pVectorField;
    DOdeSolver* pSolver;
    DTimeMap* pTimeMap;
    #pragma omp critical(capd_init)
    {
        pVectorField = new DMap(RIVAS_VECTOR_FIELD);
        pVectorField->setParameter("Z", PhysicalData::carbonProtons);
        pVectorField->setParameter("alpha", PhysicalData::alpha);
        pVectorField->setParameter("rB", PhysicalData::reducedBohr);
        pSolver = new DOdeSolver(*pVectorField, PhysicalData::capdOrder);
        pTimeMap = new DTimeMap(*pSolver);
    }
    DTimeMap& timeMap = *pTimeMap;

    // Load initial state into CAPD DVector
    DVector state(12);
    for (int i = 0; i < 12; i++) state[i] = electron.currentState[i];

    electron.storePoint();

    bool stopped = false;
    double t = 0.0;

    try {
        while (t < PhysicalData::maxTime && !stopped) {

            // Adaptive step based on distance to nucleus (charge center)
            double distCharge = std::sqrt(state[3]*state[3] + state[4]*state[4] + state[5]*state[5]);
            double divisor = (distCharge < 10.0) ? PhysicalData::capdStepDivisor : 2.0;
            double dt = distCharge / divisor;
            if (dt > PhysicalData::maxStep) dt = PhysicalData::maxStep;
            if (dt < PhysicalData::capdMinStep) dt = PhysicalData::capdMinStep;

            DVector result = timeMap(dt, state);
            state = result;
            t += dt;

            // Copy back to electron
            State s;
            for (int i = 0; i < 12; i++) s[i] = state[i];
            electron.loadState(s);
            electron.storePoint();
            if (PhysicalData::debug) electron.debugUpdate();

            // Check forward detection: qz > +1000
            if (state[2] > PhysicalData::detectionDistance) stopped = true;

            // Check backward detection: qz < -1000 and heading away (vz < 0)
            if (state[2] < -PhysicalData::detectionDistance && state[8] < 0) stopped = true;

            // Check superluminal: v^2 >= 0.9999
            double v2 = state[6]*state[6] + state[7]*state[7] + state[8]*state[8];
            if (v2 > 0.9999) { electron.isNaN = true; stopped = true; }
        }
    } catch (std::exception& e) {
        std::cerr << "CAPD exception: " << e.what() << "\n";
        electron.isNaN = true;
    } catch (...) {
        electron.isNaN = true;
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
SimulationResult runBoost(double rangeMin, double rangeMax, std::mt19937& rng) {

    using namespace boost::numeric::odeint;
    typedef runge_kutta_dopri5<State> dopri5_type;
    typedef controlled_runge_kutta<dopri5_type> controlled_type;

    Electron electron(PhysicalData::startEnergy, rangeMin, rangeMax, rng);
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
            if (PhysicalData::debug) electron.debugUpdate();

            // Check forward detection: qz > +1000
            if (state[QZ] > PhysicalData::detectionDistance) stopped = true;

            // Check backward detection: qz < -1000 and heading away (vz < 0)
            if (state[QZ] < -PhysicalData::detectionDistance && state[VZ] < 0) stopped = true;

            // Check superluminal: v^2 >= 0.9999
            double v2 = state[VX]*state[VX] + state[VY]*state[VY] + state[VZ]*state[VZ];
            if (v2 > 0.9999) { electron.isNaN = true; stopped = true; }
        }
    } catch (std::exception& e) {
        std::cerr << "Boost exception: " << e.what() << "\n";
        electron.isNaN = true;
    } catch (...) {
        electron.isNaN = true;
    }

    electron.loadState(state);

    auto endTime = std::chrono::steady_clock::now();
    long ms = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    return {std::move(electron), ms};
}

// ============================================================================
// Dispatcher — compile-time integrator selection
// ============================================================================
SimulationResult runSingleSimulation(double rangeMin, double rangeMax, std::mt19937& rng) {
    if constexpr (PhysicalData::integrator == PhysicalData::Integrator::CAPD) {
#if HAVE_CAPD
        return runCapd(rangeMin, rangeMax, rng);
#else
        static_assert(false, "CAPD selected but capd/capdlib.h not found. Install CAPD or switch to Boost.");
#endif
    } else {
        return runBoost(rangeMin, rangeMax, rng);
    }
}

// ============================================================================
// Main
// ============================================================================
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

    if constexpr (PhysicalData::integrator == PhysicalData::Integrator::CAPD) {
        std::cout << "Integrator: CAPD Taylor (order " << PhysicalData::capdOrder << ")"
                  << " | stepDivisor: " << PhysicalData::capdStepDivisor << "\n";
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
