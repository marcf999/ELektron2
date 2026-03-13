import java.text.DecimalFormat;
import java.util.ArrayDeque;

public class Electron {
    // Camera captures every Nth point while electron is within the atom chain z-range
    private static final int CAMERA_DECIMATION = 200;  // for Java DP853
    private static final double CAMERA_Z_MIN = PhysicalData.atomZ[0] - 2.0 * PhysicalData.atomSpacing;
    private static final double CAMERA_Z_MAX = PhysicalData.atomZ[PhysicalData.atomCount - 1] + 2.0 * PhysicalData.atomSpacing;
    private static final DecimalFormat FMT = new DecimalFormat("#.######E0");

    // State vector indices (matches CAPD variable order: q1,q2,q3,r1,r2,r3,v1,v2,v3,u1,u2,u3)
    public static final int QX = 0, QY = 1, QZ = 2;   // center of mass position
    public static final int RX = 3, RY = 4, RZ = 5;   // center of charge position
    public static final int VX = 6, VY = 7, VZ = 8;   // center of mass velocity
    public static final int UX = 9, UY = 10, UZ = 11;  // center of charge velocity

    public double[] electronCurrentState = new double[12];
    public double[] electronInitialState = new double[12];

    public ArrayDeque<double[]> electronStateHistory;
    public ArrayDeque<double[]> electronStateCamera;

    public double initialKineticEnergy;

    public boolean recordCamera = false;  // only true for electrons that need visualization
    public int internalCount = 0;
    private int cameraCounter = 0;

    // Initial position (d tilde in rivas notes)
    double dxZERO, dyZERO, dzZERO;
    // Initial spin
    double theta0, phi0;
    // Initial phase of zitter
    double psi0;

    public Electron(double initialKineticEnergy, double rangeMin, double rangeMax) {

        electronStateHistory = new ArrayDeque<>();
        electronStateCamera = new ArrayDeque<>();

        // Generate random impact parameter, reduce it
        dxZERO = rangeMin + Math.random() * (rangeMax - rangeMin);
        if (Math.random() > 0.5d) {
            dxZERO = -dxZERO;
        }
        dxZERO = dxZERO / PhysicalData.zitterRadius;
        dyZERO = 0;
        dzZERO = PhysicalData.startPos;

        // Calculate initial velocity from kinetic energy
        this.initialKineticEnergy = initialKineticEnergy;
        double gamma0 = initialKineticEnergy / PhysicalData.m0c2 + 1d;
        double beta0 = Math.sqrt(1 - 1 / (gamma0 * gamma0));
        double velocity0 = beta0;

        double Xdotx0 = 0;
        double Xdoty0 = 0;
        double Xdotz0 = velocity0;

        // Initialize spin along z-axis (axis of propagation)
        theta0 = (PhysicalData.spin >= 0) ? 0.0 : Math.PI;
        phi0 = 0.0;

        // Random zitter phase
        psi0 = Math.random() * 2 * Math.PI;

        // Calculate zitter position from Rivas (r tilde zero)
        double rxZERO = Math.cos(theta0) * Math.cos(phi0) * Math.cos(psi0) - Math.sin(phi0) * Math.sin(psi0);
        double ryZERO = Math.cos(theta0) * Math.sin(phi0) * Math.cos(psi0) + Math.cos(phi0) * Math.sin(psi0);
        double rzZERO = -Math.sin(theta0) * Math.cos(psi0);

        // Calculate zitter velocity from Rivas (u tilde zero)
        double uxZERO = Math.cos(theta0) * Math.cos(phi0) * Math.sin(psi0) + Math.sin(phi0) * Math.cos(psi0);
        double uyZERO = Math.cos(theta0) * Math.sin(phi0) * Math.sin(psi0) - Math.cos(phi0) * Math.cos(psi0);
        double uzZERO = -Math.sin(theta0) * Math.sin(psi0);

        // Dot products for boost
        double vdotrZero = (Xdotx0 * rxZERO) + (Xdoty0 * ryZERO) + (Xdotz0 * rzZERO);
        double vdotuZero = (Xdotx0 * uxZERO) + (Xdoty0 * uyZERO) + (Xdotz0 * uzZERO);

        // q(0) = boosted mass position
        double Xx0 = (vdotrZero * uxZERO) - (vdotuZero * rxZERO) + dxZERO;
        double Xy0 = (vdotrZero * uyZERO) - (vdotuZero * ryZERO) + dyZERO;
        double Xz0 = (vdotrZero * uzZERO) - (vdotuZero * rzZERO) + dzZERO;

        // r(0) = boosted charge position
        double Zx0 = rxZERO - (gamma0 / (1 + gamma0)) * vdotrZero * Xdotx0 + dxZERO;
        double Zy0 = ryZERO - (gamma0 / (1 + gamma0)) * vdotrZero * Xdoty0 + dyZERO;
        double Zz0 = rzZERO - (gamma0 / (1 + gamma0)) * vdotrZero * Xdotz0 + dzZERO;

        // u(0) = boosted charge velocity
        double Zdotx0 = (uxZERO + gamma0 * Xdotx0 + ((gamma0 * gamma0) / (1 + gamma0)) * vdotuZero * Xdotx0) /
                (gamma0 * (1 + vdotuZero));
        double Zdoty0 = (uyZERO + gamma0 * Xdoty0 + ((gamma0 * gamma0) / (1 + gamma0)) * vdotuZero * Xdoty0) /
                (gamma0 * (1 + vdotuZero));
        double Zdotz0 = (uzZERO + gamma0 * Xdotz0 + ((gamma0 * gamma0) / (1 + gamma0)) * vdotuZero * Xdotz0) /
                (gamma0 * (1 + vdotuZero));

        electronCurrentState[QX] = Xx0;
        electronCurrentState[QY] = Xy0;
        electronCurrentState[QZ] = Xz0;
        electronCurrentState[RX] = Zx0;
        electronCurrentState[RY] = Zy0;
        electronCurrentState[RZ] = Zz0;
        electronCurrentState[VX] = Xdotx0;
        electronCurrentState[VY] = Xdoty0;
        electronCurrentState[VZ] = Xdotz0;
        electronCurrentState[UX] = Zdotx0;
        electronCurrentState[UY] = Zdoty0;
        electronCurrentState[UZ] = Zdotz0;

        if (PhysicalData.logInitialConditions) {
            System.out.println(" INITIAL CONDITIONS | Energy:" + initialKineticEnergy + "eV | Theta0:" + theta0 * 360 / (2 * Math.PI) + " | Phi0:" + phi0 * 360 / (2 * Math.PI) + " | psi0:" + psi0 * 360 / (2 * Math.PI));
        }

        electronStateHistory.add(electronCurrentState.clone());
        electronInitialState = electronCurrentState.clone();
    }

    public void loadState(double[] state) {
        System.arraycopy(state, 0, electronCurrentState, 0, 12);
    }

    public void storePoint() {
        if (recordCamera && electronCurrentState[QZ] >= CAMERA_Z_MIN && electronCurrentState[QZ] <= CAMERA_Z_MAX) {
            if (++cameraCounter >= CAMERA_DECIMATION) {
                cameraCounter = 0;
                electronStateCamera.add(electronCurrentState.clone());
            }
        }
        electronStateHistory.add(electronCurrentState.clone());
        if (electronStateHistory.size() > 1000) electronStateHistory.removeFirst();
        internalCount++;
    }

    public double getXminusZ2() {
        double dx = electronCurrentState[QX] - electronCurrentState[RX];
        double dy = electronCurrentState[QY] - electronCurrentState[RY];
        double dz = electronCurrentState[QZ] - electronCurrentState[RZ];
        return dx * dx + dy * dy + dz * dz;
    }

    public double getXdot2() {
        return electronCurrentState[VX] * electronCurrentState[VX] +
                electronCurrentState[VY] * electronCurrentState[VY] +
                electronCurrentState[VZ] * electronCurrentState[VZ];
    }

    public double getZdot2() {
        return electronCurrentState[UX] * electronCurrentState[UX] +
                electronCurrentState[UY] * electronCurrentState[UY] +
                electronCurrentState[UZ] * electronCurrentState[UZ];
    }

    public double getGamma() {
        double v2 = getXdot2();
        if (v2 > 1) return 1E6d;
        return 1 / Math.sqrt(1 - v2);
    }

    public double getKineticEnergy() {
        return (getGamma() - 1d) * PhysicalData.m0c2;
    }

    public double getAngle() {
        double angleInRadians = Math.atan2(electronCurrentState[QZ], electronCurrentState[QX]);
        double angleInDegrees = Math.toDegrees(angleInRadians);
        if (angleInDegrees < 0) angleInDegrees += 360;
        return angleInDegrees;
    }

    public String format(double number) { return FMT.format(number); }

    public String getEXIT() {
        return " | Start position: " + format(electronStateHistory.getFirst()[QX]) +
                " | Finish position: " + format(electronStateHistory.getLast()[QX]) +
                " | Angle out: " + (int) (getAngle()) +
                "deg | Energy out: " + format(getKineticEnergy()) + "eV";
    }

    public String printState() {
        return " |Xx: " + format(electronCurrentState[QX]) +
                " | Xy: " + format(electronCurrentState[QY]) +
                " | Xz: " + format(electronCurrentState[QZ]) +
                " | Zx: " + format(electronCurrentState[RX]) +
                " | Zy: " + format(electronCurrentState[RY]) +
                " | Zz: " + format(electronCurrentState[RZ]) +
                " | Xdotx: " + format(electronCurrentState[VX]) +
                " | Xdoty: " + format(electronCurrentState[VY]) +
                " | Xdotz: " + format(electronCurrentState[VZ]) +
                " | Zdotx: " + format(electronCurrentState[UX]) +
                " | Zdoty: " + format(electronCurrentState[UY]) +
                " | Zdotz: " + format(electronCurrentState[UZ]) +
                " | Xdot2: " + format(getXdot2()) +
                " | Zdot2: " + format(getZdot2()) +
                " | RadiusX-Z: " + format(getXminusZ2()) +
                " | Energy: " + format(getKineticEnergy()) + "eV" +
                " | Gamma: " + format(getGamma());
    }
}
