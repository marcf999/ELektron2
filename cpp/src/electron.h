#pragma once

#include "physical_data.h"
#include <array>
#include <deque>
#include <cmath>
#include <string>
#include <sstream>
#include <iomanip>
#include <random>

// State vector indices
enum StateIdx {
    QX = 0, QY = 1, QZ = 2,   // center of mass position
    RX = 3, RY = 4, RZ = 5,   // center of charge position
    VX = 6, VY = 7, VZ = 8,   // center of mass velocity
    UX = 9, UY = 10, UZ = 11  // center of charge velocity
};

using State = std::array<double, 12>;

struct Electron {
    // Camera captures every Nth point while electron is within the atom chain z-range
    static constexpr int CAMERA_DECIMATION = 200;  // for Boost (~13.7M steps)
    // Chain z-range for camera capture: first atom to last atom with margin
    static inline const double CAMERA_Z_MIN = PhysicalData::atomZ[0] - 2.0 * PhysicalData::atomSpacing;
    static inline const double CAMERA_Z_MAX = PhysicalData::atomZ[PhysicalData::atomCount - 1] + 2.0 * PhysicalData::atomSpacing;

    State currentState{};
    State initialState{};

    std::deque<State> stateHistory;
    std::vector<State> stateCamera;

    double initialKineticEnergy = 0;

    bool recordCamera = false;
    int internalCount = 0;
    int cameraCounter = 0;

    // Initial conditions
    double dxZERO = 0, dyZERO = 0, dzZERO = 0;
    double theta0 = 0, phi0 = 0, psi0 = 0;

    Electron() = default;

    Electron(double energy, double rangeMin, double rangeMax, std::mt19937& rng) {
        std::uniform_real_distribution<double> impactDist(rangeMin, rangeMax);
        std::uniform_real_distribution<double> signDist(0.0, 1.0);
        std::uniform_real_distribution<double> phaseDist(0.0, 2.0 * M_PI);

        // Generate random impact parameter, reduce it
        dxZERO = impactDist(rng);
        if (signDist(rng) > 0.5) dxZERO = -dxZERO;
        dxZERO /= PhysicalData::zitterRadius;
        dyZERO = 0.0;
        dzZERO = PhysicalData::startPos;

        // Calculate initial velocity from kinetic energy
        initialKineticEnergy = energy;
        double gamma0 = energy / PhysicalData::m0c2 + 1.0;
        double beta0 = std::sqrt(1.0 - 1.0 / (gamma0 * gamma0));
        double velocity0 = beta0;

        double Xdotx0 = 0.0;
        double Xdoty0 = 0.0;
        double Xdotz0 = velocity0;

        // Initialize spin axis — uniform random on sphere if "random", else fixed
        if (PhysicalData::spinRandom) {
            std::uniform_real_distribution<double> uni01(0.0, 1.0);
            theta0 = std::acos(1.0 - 2.0 * uni01(rng));
            phi0   = 2.0 * M_PI * uni01(rng);
        } else {
            theta0 = PhysicalData::spinTheta0;
            phi0   = PhysicalData::spinPhi0;
        }

        // Random zitter phase
        psi0 = phaseDist(rng);

        // Calculate zitter position from Rivas (r tilde zero)
        double rxZERO, ryZERO, rzZERO;
        double uxZERO, uyZERO, uzZERO;

        if (PhysicalData::noZitter) {
            // No zitter: charge locked to mass (classical point particle)
            rxZERO = ryZERO = rzZERO = 0.0;
            uxZERO = uyZERO = uzZERO = 0.0;
        } else {
            rxZERO = std::cos(theta0) * std::cos(phi0) * std::cos(psi0) - std::sin(phi0) * std::sin(psi0);
            ryZERO = std::cos(theta0) * std::sin(phi0) * std::cos(psi0) + std::cos(phi0) * std::sin(psi0);
            rzZERO = -std::sin(theta0) * std::cos(psi0);

            // Calculate zitter velocity from Rivas (u tilde zero)
            uxZERO = std::cos(theta0) * std::cos(phi0) * std::sin(psi0) + std::sin(phi0) * std::cos(psi0);
            uyZERO = std::cos(theta0) * std::sin(phi0) * std::sin(psi0) - std::cos(phi0) * std::cos(psi0);
            uzZERO = -std::sin(theta0) * std::sin(psi0);
        }

        // Dot products for boost
        double vdotrZero = Xdotx0 * rxZERO + Xdoty0 * ryZERO + Xdotz0 * rzZERO;
        double vdotuZero = Xdotx0 * uxZERO + Xdoty0 * uyZERO + Xdotz0 * uzZERO;

        // q(0) = boosted mass position
        double Xx0 = vdotrZero * uxZERO - vdotuZero * rxZERO + dxZERO;
        double Xy0 = vdotrZero * uyZERO - vdotuZero * ryZERO + dyZERO;
        double Xz0 = vdotrZero * uzZERO - vdotuZero * rzZERO + dzZERO;

        // r(0) = boosted charge position
        double Zx0 = rxZERO - (gamma0 / (1.0 + gamma0)) * vdotrZero * Xdotx0 + dxZERO;
        double Zy0 = ryZERO - (gamma0 / (1.0 + gamma0)) * vdotrZero * Xdoty0 + dyZERO;
        double Zz0 = rzZERO - (gamma0 / (1.0 + gamma0)) * vdotrZero * Xdotz0 + dzZERO;

        // u(0) = boosted charge velocity
        double Zdotx0 = (uxZERO + gamma0 * Xdotx0 + (gamma0 * gamma0 / (1.0 + gamma0)) * vdotuZero * Xdotx0)
                       / (gamma0 * (1.0 + vdotuZero));
        double Zdoty0 = (uyZERO + gamma0 * Xdoty0 + (gamma0 * gamma0 / (1.0 + gamma0)) * vdotuZero * Xdoty0)
                       / (gamma0 * (1.0 + vdotuZero));
        double Zdotz0 = (uzZERO + gamma0 * Xdotz0 + (gamma0 * gamma0 / (1.0 + gamma0)) * vdotuZero * Xdotz0)
                       / (gamma0 * (1.0 + vdotuZero));

        currentState[QX] = Xx0;  currentState[QY] = Xy0;  currentState[QZ] = Xz0;
        currentState[RX] = Zx0;  currentState[RY] = Zy0;  currentState[RZ] = Zz0;
        currentState[VX] = Xdotx0; currentState[VY] = Xdoty0; currentState[VZ] = Xdotz0;
        currentState[UX] = Zdotx0; currentState[UY] = Zdoty0; currentState[UZ] = Zdotz0;

        initialState = currentState;
        stateHistory.push_back(currentState);
    }

    void loadState(const State& state) { currentState = state; }

    void storePoint() {
        if (recordCamera && currentState[QZ] >= CAMERA_Z_MIN && currentState[QZ] <= CAMERA_Z_MAX) {
            if (++cameraCounter >= CAMERA_DECIMATION) {
                cameraCounter = 0;
                stateCamera.push_back(currentState);
            }
        }
        stateHistory.push_back(currentState);
        if (stateHistory.size() > 1000) stateHistory.pop_front();
        internalCount++;
    }

    double getXminusZ2() const {
        double dx = currentState[QX] - currentState[RX];
        double dy = currentState[QY] - currentState[RY];
        double dz = currentState[QZ] - currentState[RZ];
        return dx*dx + dy*dy + dz*dz;
    }

    double getXdot2() const {
        return currentState[VX]*currentState[VX] +
               currentState[VY]*currentState[VY] +
               currentState[VZ]*currentState[VZ];
    }

    double getZdot2() const {
        return currentState[UX]*currentState[UX] +
               currentState[UY]*currentState[UY] +
               currentState[UZ]*currentState[UZ];
    }

    double getGamma() const {
        double v2 = getXdot2();
        if (v2 > 1.0) return 1e6;
        return 1.0 / std::sqrt(1.0 - v2);
    }

    double getKineticEnergy() const { return (getGamma() - 1.0) * PhysicalData::m0c2; }

    double getAngle() const {
        double a = std::atan2(currentState[QZ], currentState[QX]) * 180.0 / M_PI;
        if (a < 0) a += 360.0;
        return a;
    }

    static std::string fmt(double v) {
        std::ostringstream oss;
        oss << std::scientific << std::setprecision(6) << v;
        return oss.str();
    }

    std::string getEXIT() {
        return " | Start position: " + fmt(stateHistory.front()[QX]) +
               " | Finish position: " + fmt(stateHistory.back()[QX]) +
               " | Angle out: " + std::to_string((int)getAngle()) +
               "deg | Energy out: " + fmt(getKineticEnergy()) + "eV";
    }
};
