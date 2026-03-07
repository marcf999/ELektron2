import org.apache.commons.math3.ode.FirstOrderDifferentialEquations;

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
 *   dv/dt = 2*Z*alpha * (r - v*(r.v)) / |r|^3 * sqrt(1 - v^2) * exp(-|r|/rB)
 *   du/dt = (q - r) * (1 - v.u) / |q - r|^2
 */
public class RivasEquations implements FirstOrderDifferentialEquations {

    private final double Z;
    private final double alpha;
    private final double rB;

    public RivasEquations() {
        this.Z = PhysicalData.carbonProtons;
        this.alpha = PhysicalData.alpha;
        this.rB = PhysicalData.reducedBohr;
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

        // dv/dt: screened Coulomb force on center of mass
        double rNorm2 = r1 * r1 + r2 * r2 + r3 * r3;
        double rNorm = Math.sqrt(rNorm2);
        double rNorm3 = rNorm2 * rNorm;

        double v2sq = v1 * v1 + v2 * v2 + v3 * v3;
        double sqrtFactor = Math.sqrt(Math.max(1.0 - v2sq, 0.0));

        double rdotv = r1 * v1 + r2 * v2 + r3 * v3;
        double screening = Math.exp(-rNorm / rB);

        if (rNorm3 > 1e-30) {
            double emFactor = 2.0 * Z * alpha * screening * sqrtFactor / rNorm3;
            yDot[6] = -emFactor * (r1 - v1 * rdotv);
            yDot[7] = -emFactor * (r2 - v2 * rdotv);
            yDot[8] = -emFactor * (r3 - v3 * rdotv);
        } else {
            yDot[6] = 0;
            yDot[7] = 0;
            yDot[8] = 0;
        }

        // du/dt: zitter constraint
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
