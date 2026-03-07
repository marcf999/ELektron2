import org.apache.commons.math3.ode.FirstOrderIntegrator;
import org.apache.commons.math3.ode.nonstiff.DormandPrince853Integrator;
import org.apache.commons.math3.ode.sampling.StepHandler;
import org.apache.commons.math3.ode.sampling.StepInterpolator;
import org.apache.commons.math3.ode.events.EventHandler;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

public class Main {

    private static final DecimalFormat FMT = new DecimalFormat("#.###E0");
    private static final int IN_FLIGHT_PER_CORE = 4;

    private static class SimulationResult {
        Electron electron;
        long elapsedTimeMs;
    }

    public static void main(String[] args) {

        int totalSimulations = PhysicalData.totalSimulations;
        int plotsToShow = PhysicalData.plotsToShow;
        int cores = Runtime.getRuntime().availableProcessors();

        int isNaN = 0, isRenorm = 0, isNeg = 0, isPos = 0, is120L = 0, is120R = 0;
        double rangeMin = PhysicalData.rangeMin, rangeMax = PhysicalData.rangeMax;

        System.out.println("PARAMS | rangeMin: " + rangeMin + " | rangeMax: " + rangeMax +
                " | startEnergy: " + PhysicalData.startEnergy + " | spin: " + PhysicalData.spin +
                " | carbonProtons(Z): " + PhysicalData.carbonProtons);
        System.out.println("Integrator: DormandPrince853 | relTol: " + PhysicalData.relTol +
                " | absTol: " + PhysicalData.absTol);
        System.out.println("Running " + totalSimulations + " simulations on " + cores + " cores.");

        ExecutorService executor = Executors.newFixedThreadPool(cores);
        CompletionService<SimulationResult> completionService = new ExecutorCompletionService<>(executor);

        int maxInFlight = Math.max(cores * IN_FLIGHT_PER_CORE, 1);
        int submitted = 0;
        int completedCount = 0;

        // Pre-submit up to maxInFlight tasks
        while (submitted < totalSimulations && submitted < maxInFlight) {
            submitTask(completionService, rangeMin, rangeMax);
            submitted++;
        }

        List<Electron> visualizationElectrons = new ArrayList<>();
        List<SimulationResult> allResults = new ArrayList<>();
        long totalStartMs = System.currentTimeMillis();

        try {
            while (completedCount < totalSimulations) {
                Future<SimulationResult> future = completionService.take();
                SimulationResult result = future.get();
                completedCount++;

                Electron electron = result.electron;
                allResults.add(result);

                if (visualizationElectrons.size() < plotsToShow) {
                    visualizationElectrons.add(electron);
                }

                // Tally state
                String stateString = "";
                if (electron.isNaN()) {
                    stateString = "isNaN";
                    isNaN++;
                } else {
                    if (electron.isPos()) { stateString = "isPos"; isPos++; }
                    if (electron.isNeg()) { stateString = "isNeg"; isNeg++; }
                    if (electron.is120R()) { stateString = "is120R"; is120R++; }
                    if (electron.is120L()) { stateString = "is120L"; is120L++; }
                    if (electron.isRenorm()) { isRenorm++; stateString += "_isRenorm"; }
                    else { stateString += "_isOK"; }
                }

                if (completedCount % PhysicalData.progressLogEvery == 0 || completedCount == totalSimulations) {
                    System.out.println(
                            "RUNS FINISHED: " + completedCount +
                                    " | STATE: " + stateString +
                                    " | isNaN: " + isNaN +
                                    " | isRenorm: " + isRenorm +
                                    " | isNeg: " + isNeg +
                                    " | isPos: " + isPos +
                                    " | is120L: " + is120L +
                                    " | is120R: " + is120R +
                                    electron.getEXIT() +
                                    electron.getConstraints() +
                                    " | Steps: " + electron.internalCount +
                                    " | Time: " + result.elapsedTimeMs + "ms"
                    );
                }

                // Submit next task if more remain
                if (submitted < totalSimulations) {
                    submitTask(completionService, rangeMin, rangeMax);
                    submitted++;
                }
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Simulation interrupted", e);
        } catch (ExecutionException e) {
            throw new RuntimeException("Simulation failed", e);
        } finally {
            executor.shutdown();
        }

        long totalElapsedMs = System.currentTimeMillis() - totalStartMs;
        System.out.println("TOTAL TIME FOR " + totalSimulations + " SIMULATIONS: " + totalElapsedMs + "ms (" + cores + " cores)");

        for (Electron electron : visualizationElectrons) {
            new PlotDots(electron);
        }

        // Write full-precision results file for ALL electrons
        writeResultsFile(allResults, totalSimulations, cores, totalElapsedMs,
                isNaN, isPos, isNeg, is120L, is120R, isRenorm);

        // Write trajectory camera data for the first plotsToShow electrons
        writeTrajectoriesFile(visualizationElectrons, totalSimulations, cores, totalElapsedMs);
    }

    private static void writeResultsFile(List<SimulationResult> allResults,
            int totalSimulations, int cores, long totalElapsedMs,
            int isNaN, int isPos, int isNeg, int is120L, int is120R, int isRenorm) {
        LocalDateTime now = LocalDateTime.now();
        String timestamp = now.format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));
        String datePart = now.format(DateTimeFormatter.ofPattern("yyyy-MM-dd"));
        String timePart = now.format(DateTimeFormatter.ofPattern("HHmmss"));

        // results/<date>_<time>_java-dp853_<iterations>.dat
        // Resolve relative to project root (parent of src/)
        File resultsDir = new File(System.getProperty("user.dir")).toPath()
                .resolve("results").toFile();
        if (!resultsDir.exists()) {
            // Try one level up (if running from src/)
            resultsDir = new File(System.getProperty("user.dir")).toPath()
                    .resolve("../results").normalize().toFile();
        }
        if (!resultsDir.exists()) resultsDir.mkdirs();
        String resultsFile = datePart + "_" + timePart + "_java-dp853_" + totalSimulations + ".dat";
        File resultsPath = new File(resultsDir, resultsFile);

        try (PrintWriter out = new PrintWriter(new FileWriter(resultsPath))) {

            // Context header
            out.println("# ELektron2 Java Simulation Results");
            out.println("# Date: " + timestamp);
            out.println("# Integrator: DormandPrince853 (commons-math3)");
            out.println("# DP853 minStep: " + repr(PhysicalData.minStep)
                    + "  maxStep: " + repr(PhysicalData.maxStep)
                    + "  absTol: " + repr(PhysicalData.absTol)
                    + "  relTol: " + repr(PhysicalData.relTol));
            out.println("# Java: " + System.getProperty("java.version")
                    + "  OS: " + System.getProperty("os.name") + " " + System.getProperty("os.arch"));
            out.println("# Cores: " + cores);
            out.println("# Total time: " + totalElapsedMs + " ms");
            out.println("# Total simulations: " + totalSimulations);
            out.println("# startEnergy: " + repr(PhysicalData.startEnergy) + " eV");
            out.println("# startPos: " + repr(PhysicalData.startPos) + " (reduced)");
            out.println("# detectionDistance: " + repr(PhysicalData.detectionDistance) + " (reduced)");
            out.println("# rangeMin: " + repr(PhysicalData.rangeMin) + " m");
            out.println("# rangeMax: " + repr(PhysicalData.rangeMax) + " m");
            out.println("# spin: " + PhysicalData.spin);
            out.println("# Z: " + repr(PhysicalData.carbonProtons));
            out.println("# alpha: " + repr(PhysicalData.alpha));
            out.println("# reducedBohr: " + repr(PhysicalData.reducedBohr));
            out.println("# zitterRadius: " + repr(PhysicalData.zitterRadius) + " m");
            out.println("# maxTime: " + repr(PhysicalData.maxTime) + " (reduced)");
            out.println("# Summary: isNaN=" + isNaN + " isPos=" + isPos
                    + " isNeg=" + isNeg + " is120L=" + is120L
                    + " is120R=" + is120R + " isRenorm=" + isRenorm);
            out.println("#");
            out.println("# Columns:");
            out.println("# idx qx qy qz rx ry rz vx vy vz ux uy uz"
                    + " energyIn_eV energyOut_eV angle_deg steps"
                    + " apexCharge apexMass v2 u2 |q-r|2"
                    + " minZdot2 maxZdot2 minXdot2 maxXdot2 minR maxR"
                    + " isNaN isPos isNeg elapsedMs"
                    + " dxZERO_reduced psi0");
            out.println("#");

            for (int i = 0; i < allResults.size(); i++) {
                SimulationResult r = allResults.get(i);
                Electron e = r.electron;
                double[] s = e.electronCurrentState;
                double v2 = s[Electron.VX]*s[Electron.VX] + s[Electron.VY]*s[Electron.VY] + s[Electron.VZ]*s[Electron.VZ];
                double u2 = s[Electron.UX]*s[Electron.UX] + s[Electron.UY]*s[Electron.UY] + s[Electron.UZ]*s[Electron.UZ];
                double dx = s[Electron.QX]-s[Electron.RX], dy = s[Electron.QY]-s[Electron.RY], dz = s[Electron.QZ]-s[Electron.RZ];
                double qr2 = dx*dx + dy*dy + dz*dz;

                out.print(i);
                for (int j = 0; j < 12; j++) out.print(" " + repr(s[j]));
                out.print(" " + repr(e.initialKineticEnergy));
                out.print(" " + repr(e.getKineticEnergy()));
                out.print(" " + repr(e.getAngle()));
                out.print(" " + e.internalCount);
                out.print(" " + repr(e.minimalDistance));
                out.print(" " + repr(e.minimalMassDistance));
                out.print(" " + repr(v2));
                out.print(" " + repr(u2));
                out.print(" " + repr(qr2));
                out.print(" " + repr(e.minZelv2) + " " + repr(e.maxZelv2));
                out.print(" " + repr(e.minXdot2) + " " + repr(e.maxXdot2));
                out.print(" " + repr(e.minR) + " " + repr(e.maxR));
                out.print(" " + (e.isNaN() ? 1 : 0));
                out.print(" " + (e.isPos() ? 1 : 0));
                out.print(" " + (e.isNeg() ? 1 : 0));
                out.print(" " + r.elapsedTimeMs);
                out.print(" " + repr(e.dxZERO));
                out.print(" " + repr(e.psi0));
                out.println();
            }

            System.out.println("Wrote " + allResults.size() + " electron results to " + resultsPath.getPath());
        } catch (Exception ex) {
            System.err.println("Failed to write " + resultsPath.getPath() + ": " + ex.getMessage());
        }
    }

    private static void writeTrajectoriesFile(List<Electron> electrons,
            int totalSimulations, int cores, long totalElapsedMs) {
        LocalDateTime now = LocalDateTime.now();
        String timestamp = now.format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));
        String datePart = now.format(DateTimeFormatter.ofPattern("yyyy-MM-dd"));
        String timePart = now.format(DateTimeFormatter.ofPattern("HHmmss"));

        File resultsDir = new File(System.getProperty("user.dir")).toPath()
                .resolve("results").toFile();
        if (!resultsDir.exists()) {
            resultsDir = new File(System.getProperty("user.dir")).toPath()
                    .resolve("../results").normalize().toFile();
        }
        if (!resultsDir.exists()) resultsDir.mkdirs();
        String trajFile = datePart + "_" + timePart + "_java-dp853_trajectories_" + totalSimulations + ".dat";
        File trajPath = new File(resultsDir, trajFile);

        try (PrintWriter out = new PrintWriter(new FileWriter(trajPath))) {

            out.println("# ELektron2 Java Trajectory Camera Data");
            out.println("# Date: " + timestamp);
            out.println("# Integrator: DormandPrince853 (commons-math3)");
            out.println("# DP853 minStep: " + repr(PhysicalData.minStep)
                    + "  maxStep: " + repr(PhysicalData.maxStep)
                    + "  absTol: " + repr(PhysicalData.absTol)
                    + "  relTol: " + repr(PhysicalData.relTol));
            out.println("# Java: " + System.getProperty("java.version")
                    + "  OS: " + System.getProperty("os.name") + " " + System.getProperty("os.arch"));
            out.println("# Cores: " + cores);
            out.println("# Total time: " + totalElapsedMs + " ms");
            out.println("# Total simulations: " + totalSimulations);
            out.println("# Trajectories in file: " + electrons.size());
            out.println("# startEnergy: " + repr(PhysicalData.startEnergy) + " eV");
            out.println("# startPos: " + repr(PhysicalData.startPos) + " (reduced)");
            out.println("# detectionDistance: " + repr(PhysicalData.detectionDistance) + " (reduced)");
            out.println("# rangeMin: " + repr(PhysicalData.rangeMin) + " m");
            out.println("# rangeMax: " + repr(PhysicalData.rangeMax) + " m");
            out.println("# spin: " + PhysicalData.spin);
            out.println("# Z: " + repr(PhysicalData.carbonProtons));
            out.println("# alpha: " + repr(PhysicalData.alpha));
            out.println("# reducedBohr: " + repr(PhysicalData.reducedBohr));
            out.println("# zitterRadius: " + repr(PhysicalData.zitterRadius) + " m");
            out.println("# maxTime: " + repr(PhysicalData.maxTime) + " (reduced)");
            out.println("# Camera threshold: distance < 100 reduced units");
            out.println("# Max camera points per electron: 20000");
            out.println("#");
            out.println("# Each trajectory block starts with:");
            out.println("#   >> TRAJECTORY idx <n> points <p> dxZERO <dx> psi0 <psi> energyIn <eIn> energyOut <eOut> angle <a>");
            out.println("# followed by per-step rows, and ends with:");
            out.println("#   << END TRAJECTORY idx <n>");
            out.println("#");
            out.println("# Columns: qx qy qz rx ry rz vx vy vz ux uy uz");
            out.println("#");

            for (int i = 0; i < electrons.size(); i++) {
                Electron e = electrons.get(i);
                out.println(">> TRAJECTORY idx " + i
                        + " points " + e.electronStateCamera.size()
                        + " dxZERO " + repr(e.dxZERO)
                        + " psi0 " + repr(e.psi0)
                        + " energyIn " + repr(e.initialKineticEnergy)
                        + " energyOut " + repr(e.getKineticEnergy())
                        + " angle " + repr(e.getAngle()));
                for (double[] s : e.electronStateCamera) {
                    out.print(repr(s[Electron.QX]));
                    out.print(" " + repr(s[Electron.QY]));
                    out.print(" " + repr(s[Electron.QZ]));
                    out.print(" " + repr(s[Electron.RX]));
                    out.print(" " + repr(s[Electron.RY]));
                    out.print(" " + repr(s[Electron.RZ]));
                    out.print(" " + repr(s[Electron.VX]));
                    out.print(" " + repr(s[Electron.VY]));
                    out.print(" " + repr(s[Electron.VZ]));
                    out.print(" " + repr(s[Electron.UX]));
                    out.print(" " + repr(s[Electron.UY]));
                    out.print(" " + repr(s[Electron.UZ]));
                    out.println();
                }
                out.println("<< END TRAJECTORY idx " + i);
                out.println("#");
            }

            System.out.println("Wrote " + electrons.size() + " trajectories to " + trajPath.getPath());
        } catch (Exception ex) {
            System.err.println("Failed to write " + trajPath.getPath() + ": " + ex.getMessage());
        }
    }

    /** Full double precision — no truncation */
    private static String repr(double v) {
        return Double.toString(v);
    }

    private static void submitTask(CompletionService<SimulationResult> cs, double rangeMin, double rangeMax) {
        cs.submit(() -> {
            Electron electron = new Electron(PhysicalData.startEnergy, rangeMin, rangeMax);
            long startMs = System.currentTimeMillis();
            runSingleSimulation(electron);
            SimulationResult result = new SimulationResult();
            result.electron = electron;
            result.elapsedTimeMs = System.currentTimeMillis() - startMs;
            return result;
        });
    }

    public static void runSingleSimulation(Electron electron) {

        RivasEquations equations = new RivasEquations();

        FirstOrderIntegrator integrator = new DormandPrince853Integrator(
                PhysicalData.minStep,
                PhysicalData.maxStep,
                PhysicalData.absTol,
                PhysicalData.relTol
        );

        // Step handler to store points for visualization
        integrator.addStepHandler(new StepHandler() {
            @Override
            public void init(double t0, double[] y0, double t) {}

            @Override
            public void handleStep(StepInterpolator interpolator, boolean isLast) {
                double[] y = interpolator.getInterpolatedState();
                electron.loadState(y);
                electron.storePoint();
                if (PhysicalData.debug) electron.debug();
            }
        });

        // Event handler: stop when electron EXITS detection sphere (qz crosses from negative to positive past detection)
        integrator.addEventHandler(new EventHandler() {
            @Override
            public void init(double t0, double[] y0, double t) {}

            @Override
            public double g(double t, double[] y) {
                return y[2] - PhysicalData.detectionDistance;
            }

            @Override
            public Action eventOccurred(double t, double[] y, boolean increasing) {
                return Action.STOP;
            }

            @Override
            public void resetState(double t, double[] y) {}
        }, 1.0, 1e-6, 100);

        // Also stop if electron goes way past in negative z (backscatter exit)
        integrator.addEventHandler(new EventHandler() {
            @Override
            public void init(double t0, double[] y0, double t) {}

            @Override
            public double g(double t, double[] y) {
                return y[2] + PhysicalData.detectionDistance;
            }

            @Override
            public Action eventOccurred(double t, double[] y, boolean increasing) {
                if (!increasing) return Action.STOP;
                return Action.CONTINUE;
            }

            @Override
            public void resetState(double t, double[] y) {}
        }, 1.0, 1e-6, 100);

        // Event handler: stop if v^2 >= 1 (superluminal)
        integrator.addEventHandler(new EventHandler() {
            @Override
            public void init(double t0, double[] y0, double t) {}

            @Override
            public double g(double t, double[] y) {
                double v2 = y[6] * y[6] + y[7] * y[7] + y[8] * y[8];
                return 0.9999 - v2;
            }

            @Override
            public Action eventOccurred(double t, double[] y, boolean increasing) {
                electron.isNaN = true;
                return Action.STOP;
            }

            @Override
            public void resetState(double t, double[] y) {}
        }, 1.0, 1e-6, 100);

        // Store initial point
        electron.storePoint();

        try {
            double[] state = electron.electronCurrentState.clone();
            integrator.integrate(equations, 0.0, state, PhysicalData.maxTime, state);
            electron.loadState(state);
        } catch (Exception e) {
            electron.isNaN = true;
        }
    }

    public static String format(double number) {
        return FMT.format(number);
    }
}
