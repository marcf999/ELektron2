import org.apache.commons.math3.ode.FirstOrderIntegrator;
import org.apache.commons.math3.ode.nonstiff.DormandPrince853Integrator;
import org.apache.commons.math3.ode.sampling.StepHandler;
import org.apache.commons.math3.ode.sampling.StepInterpolator;
import org.apache.commons.math3.ode.events.EventHandler;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

public class Main {

    private static final int IN_FLIGHT_PER_CORE = 4;

    private static class SimulationResult {
        Electron electron;
        long elapsedTimeMs;
    }

    public static void main(String[] args) {

        int totalSimulations = PhysicalData.totalSimulations;
        int plotsToShow = PhysicalData.plotsToShow;
        int cores = Runtime.getRuntime().availableProcessors();

        double rangeMin = PhysicalData.rangeMin, rangeMax = PhysicalData.rangeMax;

        System.out.println("PARAMS | rangeMin: " + rangeMin + " | rangeMax: " + rangeMax +
                " | startEnergy: " + PhysicalData.startEnergy + " | spin: " + PhysicalData.spin +
                " | Z: " + PhysicalData.carbonProtons +
                " | atoms: " + PhysicalData.atomCount);
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
            submitTask(completionService, rangeMin, rangeMax, submitted < plotsToShow);
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

                if (completedCount % PhysicalData.progressLogEvery == 0 || completedCount == totalSimulations) {
                    System.out.println(
                            "RUNS FINISHED: " + completedCount +
                                    electron.getEXIT() +
                                    " | Steps: " + electron.internalCount +
                                    " | Time: " + result.elapsedTimeMs + "ms"
                    );
                }

                // Submit next task if more remain
                if (submitted < totalSimulations) {
                    submitTask(completionService, rangeMin, rangeMax, submitted < plotsToShow);
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
            Electron e = electron;
            javax.swing.SwingUtilities.invokeLater(() -> new PlotDots(e));
        }

        // Write full-precision results file for ALL electrons
        writeResultsFile(allResults, totalSimulations, cores, totalElapsedMs);
    }

    private static void writeResultsFile(List<SimulationResult> allResults,
            int totalSimulations, int cores, long totalElapsedMs) {
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
            out.println("# atomCount: " + PhysicalData.atomCount);
            out.println("# atomSpacing: " + repr(PhysicalData.atomSpacing) + " (reduced) = "
                    + repr(PhysicalData.atomSpacingMeters) + " m");
            out.println("# chainHalfLength: " + repr(PhysicalData.chainHalfLength) + " (reduced)");
            out.println("#");
            out.println("# Columns:");
            out.println("# idx qx qy qz rx ry rz vx vy vz ux uy uz"
                    + " energyIn_eV energyOut_eV angle_deg steps"
                    + " elapsedMs dxZERO_reduced psi0");
            out.println("#");

            for (int i = 0; i < allResults.size(); i++) {
                SimulationResult r = allResults.get(i);
                Electron e = r.electron;
                double[] s = e.electronCurrentState;

                out.print(i);
                for (int j = 0; j < 12; j++) out.print(" " + repr(s[j]));
                out.print(" " + repr(e.initialKineticEnergy));
                out.print(" " + repr(e.getKineticEnergy()));
                out.print(" " + repr(e.getAngle()));
                out.print(" " + e.internalCount);
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

    /** Full double precision — no truncation */
    private static String repr(double v) {
        return Double.toString(v);
    }

    private static void submitTask(CompletionService<SimulationResult> cs, double rangeMin, double rangeMax, boolean recordCamera) {
        cs.submit(() -> {
            Electron electron = new Electron(PhysicalData.startEnergy, rangeMin, rangeMax);
            electron.recordCamera = recordCamera;
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
            }
        });

        // Event handler: stop when electron passes forward detection boundary
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

        // Event handler: stop on backscatter exit (negative z, decreasing)
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

        // Event handler: stop if |qx| exceeds 10 Bohr radii (positive side)
        integrator.addEventHandler(new EventHandler() {
            @Override
            public void init(double t0, double[] y0, double t) {}

            @Override
            public double g(double t, double[] y) {
                return PhysicalData.xyBoundary - y[0];
            }

            @Override
            public Action eventOccurred(double t, double[] y, boolean increasing) {
                return Action.STOP;
            }

            @Override
            public void resetState(double t, double[] y) {}
        }, 1.0, 1e-6, 100);

        // Event handler: stop if |qx| exceeds 10 Bohr radii (negative side)
        integrator.addEventHandler(new EventHandler() {
            @Override
            public void init(double t0, double[] y0, double t) {}

            @Override
            public double g(double t, double[] y) {
                return y[0] + PhysicalData.xyBoundary;
            }

            @Override
            public Action eventOccurred(double t, double[] y, boolean increasing) {
                return Action.STOP;
            }

            @Override
            public void resetState(double t, double[] y) {}
        }, 1.0, 1e-6, 100);

        // Event handler: stop if |qy| exceeds 10 Bohr radii (positive side)
        integrator.addEventHandler(new EventHandler() {
            @Override
            public void init(double t0, double[] y0, double t) {}

            @Override
            public double g(double t, double[] y) {
                return PhysicalData.xyBoundary - y[1];
            }

            @Override
            public Action eventOccurred(double t, double[] y, boolean increasing) {
                return Action.STOP;
            }

            @Override
            public void resetState(double t, double[] y) {}
        }, 1.0, 1e-6, 100);

        // Event handler: stop if |qy| exceeds 10 Bohr radii (negative side)
        integrator.addEventHandler(new EventHandler() {
            @Override
            public void init(double t0, double[] y0, double t) {}

            @Override
            public double g(double t, double[] y) {
                return y[1] + PhysicalData.xyBoundary;
            }

            @Override
            public Action eventOccurred(double t, double[] y, boolean increasing) {
                return Action.STOP;
            }

            @Override
            public void resetState(double t, double[] y) {}
        }, 1.0, 1e-6, 100);

        // Event handler: stop if v^2 >= 0.9999 (superluminal guard)
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
            System.err.println("Integration exception: " + e.getMessage());
        }
    }
}
