import org.apache.commons.math3.ode.FirstOrderDifferentialEquations;

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
public class RivasEquations implements FirstOrderDifferentialEquations {

    private final double Z;
    private final double alpha;
    private final double rB;
    private final int atomCount;
    private final double[] atomZ;

    public RivasEquations() {
        this.Z = PhysicalData.carbonProtons;
        this.alpha = PhysicalData.alpha;
        this.rB = PhysicalData.reducedBohr;
        this.atomCount = PhysicalData.atomCount;
        this.atomZ = PhysicalData.atomZ;
    }

    @Override
    public int getDimension() {
        return 12;
    }

    @Override
    public void computeDerivatives(double t, double[] y, double[] yDot) {

        double q1 = y[0], q2 = y[1], q3 = y[2];
        double r1 = y[3], r2 = y[4], r3 = y[5];
        double v1 = y[6], v2 = y[7], v3 = y[8];
        double u1 = y[9], u2 = y[10], u3 = y[11];

        // dq/dt = v
        yDot[0] = v1;
        yDot[1] = v2;
        yDot[2] = v3;

        // dr/dt = u
        yDot[3] = u1;
        yDot[4] = u2;
        yDot[5] = u3;

        // dv/dt: screened Coulomb force on center of mass — summed over all atoms
        double v2sq = v1 * v1 + v2 * v2 + v3 * v3;
        double sqrtFactor = Math.sqrt(Math.max(1.0 - v2sq, 0.0));
        double twoZAlpha = 2.0 * Z * alpha;

        double dv1 = 0, dv2 = 0, dv3 = 0;

        for (int k = 0; k < atomCount; k++) {
            // Displacement from atom k to charge center
            // Atoms are along z-axis at (0, 0, atomZ[k])
            double d1 = r1;
            double d2 = r2;
            double d3 = r3 - atomZ[k];

            double dNorm2 = d1 * d1 + d2 * d2 + d3 * d3;
            double dNorm = Math.sqrt(dNorm2);
            double dNorm3 = dNorm2 * dNorm;

            if (dNorm3 > 1e-30) {
                double ddotv = d1 * v1 + d2 * v2 + d3 * v3;
                double screening = Math.exp(-dNorm / rB);
                double emFactor = twoZAlpha * screening * sqrtFactor / dNorm3;
                dv1 -= emFactor * (d1 - v1 * ddotv);
                dv2 -= emFactor * (d2 - v2 * ddotv);
                dv3 -= emFactor * (d3 - v3 * ddotv);
            }
        }

        yDot[6] = dv1;
        yDot[7] = dv2;
        yDot[8] = dv3;

        // du/dt: zitter constraint (unchanged — depends only on q-r and v.u)
        double qr1 = q1 - r1, qr2 = q2 - r2, qr3 = q3 - r3;
        double qrNorm2 = qr1 * qr1 + qr2 * qr2 + qr3 * qr3;
        double vdotu = v1 * u1 + v2 * u2 + v3 * u3;

        if (qrNorm2 > 1e-30) {
            double zitterFactor = (1.0 - vdotu) / qrNorm2;
            yDot[9] = zitterFactor * qr1;
            yDot[10] = zitterFactor * qr2;
            yDot[11] = zitterFactor * qr3;
        } else {
            yDot[9] = 0;
            yDot[10] = 0;
            yDot[11] = 0;
        }
    }
}
