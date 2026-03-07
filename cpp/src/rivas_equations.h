#pragma once

#include "physical_data.h"
#include "electron.h"
#include <cmath>
#include <array>

/**
 * 12D Relativistic Mott-Rivas equations of motion.
 *
 * State vector: [q1,q2,q3, r1,r2,r3, v1,v2,v3, u1,u2,u3]
 *   q = center of mass position
 *   r = center of charge position
 *   v = center of mass velocity (v < c)
 *   u = center of charge velocity (|u| = c = 1)
 *
 * Equations (reduced units, c=1):
 *   dq/dt = v
 *   dr/dt = u
 *   dv/dt = -2*Z*alpha * (r - v*(r.v)) / |r|^3 * sqrt(1 - v^2) * exp(-|r|/rB)
 *   du/dt = (q - r) * (1 - v.u) / |q - r|^2
 */
struct RivasEquations {

    double Z;
    double alpha;
    double rB;

    RivasEquations()
        : Z(PhysicalData::carbonProtons),
          alpha(PhysicalData::alpha),
          rB(PhysicalData::reducedBohr) {}

    // Boost.Odeint system function: dydt = f(y, t)
    void operator()(const State& y, State& dydt, double /*t*/) const {

        double q1 = y[0], q2 = y[1], q3 = y[2];
        double r1 = y[3], r2 = y[4], r3 = y[5];
        double v1 = y[6], v2 = y[7], v3 = y[8];
        double u1 = y[9], u2 = y[10], u3 = y[11];

        // dq/dt = v
        dydt[0] = v1;
        dydt[1] = v2;
        dydt[2] = v3;

        // dr/dt = u
        dydt[3] = u1;
        dydt[4] = u2;
        dydt[5] = u3;

        // dv/dt: screened Coulomb force on center of mass
        double rNorm2 = r1*r1 + r2*r2 + r3*r3;
        double rNorm = std::sqrt(rNorm2);
        double rNorm3 = rNorm2 * rNorm;

        double v2sq = v1*v1 + v2*v2 + v3*v3;
        double sqrtFactor = std::sqrt(std::max(1.0 - v2sq, 0.0));

        double rdotv = r1*v1 + r2*v2 + r3*v3;
        double screening = std::exp(-rNorm / rB);

        if (rNorm3 > 1e-30) {
            double emFactor = 2.0 * Z * alpha * screening * sqrtFactor / rNorm3;
            dydt[6] = -emFactor * (r1 - v1 * rdotv);
            dydt[7] = -emFactor * (r2 - v2 * rdotv);
            dydt[8] = -emFactor * (r3 - v3 * rdotv);
        } else {
            dydt[6] = 0.0;
            dydt[7] = 0.0;
            dydt[8] = 0.0;
        }

        // du/dt: zitter constraint
        double qr1 = q1 - r1, qr2 = q2 - r2, qr3 = q3 - r3;
        double qrNorm2 = qr1*qr1 + qr2*qr2 + qr3*qr3;
        double vdotu = v1*u1 + v2*u2 + v3*u3;

        if (qrNorm2 > 1e-30) {
            double zitterFactor = (1.0 - vdotu) / qrNorm2;
            dydt[9]  = zitterFactor * qr1;
            dydt[10] = zitterFactor * qr2;
            dydt[11] = zitterFactor * qr3;
        } else {
            dydt[9]  = 0.0;
            dydt[10] = 0.0;
            dydt[11] = 0.0;
        }
    }
};
