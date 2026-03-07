public class PhysicalData {

    // Impact parameter range in meters
    public static double rangeMin = 1e-12d, rangeMax = 1e-10d;

    public static int spin = +1;

    // Zitter radius in m: hbar/(2mc)
    public static double zitterRadius = 1.93079634e-13d;

    // Simulation parameters
    public static double startEnergy = 5000d;       // eV
    public static double startPos = -1000d;          // reduced units (zitter radii)
    public static int totalSimulations = 1000;
    public static int plotsToShow = 10;

    // Integrator tolerances
    public static double relTol = 1e-12d;
    public static double absTol = 1e-12d;
    public static double minStep = 1e-10d;
    public static double maxStep = 10d;

    // Detection cutoff in reduced units
    public static double detectionDistance = 1000d;

    // Max integration time (reduced units)
    public static double maxTime = 1e6d;

    // Logging
    public static boolean debug = true;
    public static boolean logInitialConditions = false;
    public static int progressLogEvery = 100;

    // Integrity tolerances
    public static double radiusTolerance = 1e-1d;
    public static double zdot2Tolerance = 1e-1d;

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
}
