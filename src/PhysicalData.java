public class PhysicalData {

    // Impact parameter range in meters
    public static double rangeMin = 1e-12d, rangeMax = 1e-10d;

    public static int spin = +1;

    // Zitter radius in m: hbar/(2mc)
    public static double zitterRadius = 1.93079634e-13d;

    // Atom chain: 30 carbon atoms along z-axis, C-C bond length spacing (graphene)
    public static int atomCount = 30;
    public static double atomSpacingMeters = 1.42e-10d;  // graphene C-C bond length in meters
    public static double atomSpacing = atomSpacingMeters / zitterRadius;  // ~735.5 reduced units
    public static double chainHalfLength = (atomCount - 1) / 2.0 * atomSpacing;

    // Pre-computed atom z-positions in reduced units, centered at z=0
    public static double[] atomZ;
    static {
        atomZ = new double[atomCount];
        for (int i = 0; i < atomCount; i++) {
            atomZ[i] = (i - (atomCount - 1) / 2.0) * atomSpacing;
        }
    }

    // Simulation parameters
    public static double startEnergy = 5000d;       // eV
    public static double startPos = -(chainHalfLength + 4000);  // reduced units, well before first atom
    public static int totalSimulations = 100;
    public static int plotsToShow = 0;

    // Integrator tolerances
    public static double relTol = 1e-12d;
    public static double absTol = 1e-12d;
    public static double minStep = 1e-10d;
    public static double maxStep = 10d;

    // Detection cutoff in reduced units — beyond the atom chain
    public static double detectionDistance = chainHalfLength + 4000;

    // Max integration time (reduced units)
    public static double maxTime = 1e6d;

    // Logging
    public static boolean logInitialConditions = false;
    public static int progressLogEvery = 100;

    // Dirac frequency in s^-1
    public static double zetaFrequency = 1.55268814e21d;

    // Dirac Time in s
    public static double ZitterTime = 6.4404433e-22d;

    // Electron Rest mass in kilograms
    public static double electronRestMass = 9.109E-31d;

    // From notes of Rivas natural units A = (ehbar)/(2m2c3)
    public static double rivasA = 3.778e-19d;
    public static double coulombFactor = 8.988E+9d, electronCharge = -1.60217663E-19d, protonCharge = -electronCharge;

    // Carbon nucleus: atomic number Z = 6
    public static double carbonProtons = 6;

    // Velocity of light
    public static double c = 2.99792458E+8d;
    public static double c2 = c * c;

    // Electron rest energy in eV
    public static double m0c2 = 5.11E+5d;

    // Fine structure constant
    public static double alpha = 0.007299270072992700d;

    // Bohr Radius in meters for screening
    public static double bohrRadius = 5.3E-11d;

    // Reduced Bohr radius (in zitter radii)
    public static double reducedBohr = bohrRadius / zitterRadius; // ~275

    // XY-boundary: stop if |qx| or |qy| exceeds 10 Bohr radii (reduced units)
    public static double xyBoundary = 10.0 * reducedBohr;

    // Detector: 1m distance, 100mm × 100mm square aperture centered on z-axis
    public static double detectorDistanceM = 1.0d;      // meters
    public static double apertureHalfM = 50e-3d;        // half-width in meters (100mm side)
}
