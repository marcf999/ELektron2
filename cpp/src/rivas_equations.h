#pragma once

#include "physical_data.h"
#include "electron.h"
#include <cmath>
#include <array>

/**
 * 12D Relativistic Mott-Rivas equations of motion for a LINEAR CHAIN of atoms.
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
 *   dv/dt = SUM_k [ -2*Z*alpha * (d_k - v*(d_k.v)) / |d_k|^3 * sqrt(1 - v^2) * exp(-|d_k|/rB) ]
 *           where d_k = r - atomPosition_k  (charge center to atom k)
 *   du/dt = (q - r) * (1 - v.u) / |q - r|^2   (zitter constraint, unchanged)
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

        // dv/dt: screened Coulomb force on center of mass — summed over all atoms
        double v2sq = v1*v1 + v2*v2 + v3*v3;
        double sqrtFactor = std::sqrt(std::max(1.0 - v2sq, 0.0));
        double twoZAlpha = 2.0 * Z * alpha;

        double dv1 = 0.0, dv2 = 0.0, dv3 = 0.0;

        for (int k = 0; k < PhysicalData::atomCount; k++) {
            // Displacement from atom k to charge center
            // Atoms are along z-axis at (0, 0, atomZ[k])
            double d1 = r1;
            double d2 = r2;
            double d3 = r3 - PhysicalData::atomZ[k];

            double dNorm2 = d1*d1 + d2*d2 + d3*d3;
            double dNorm = std::sqrt(dNorm2);
            double dNorm3 = dNorm2 * dNorm;

            if (dNorm3 > 1e-30) {
                double ddotv = d1*v1 + d2*v2 + d3*v3;
                double screening = std::exp(-dNorm / rB);
                double emFactor = twoZAlpha * screening * sqrtFactor / dNorm3;
                dv1 -= emFactor * (d1 - v1 * ddotv);
                dv2 -= emFactor * (d2 - v2 * ddotv);
                dv3 -= emFactor * (d3 - v3 * ddotv);
            }
        }

        dydt[6] = dv1;
        dydt[7] = dv2;
        dydt[8] = dv3;

        // du/dt: zitter constraint (unchanged — depends only on q-r and v.u)
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
