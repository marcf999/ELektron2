import org.apache.commons.math3.ode.FirstOrderIntegrator;
import org.apache.commons.math3.ode.nonstiff.DormandPrince853Integrator;
import org.apache.commons.math3.ode.sampling.StepHandler;
import org.apache.commons.math3.ode.sampling.StepInterpolator;
import org.apache.commons.math3.ode.events.EventHandler;

import java.text.DecimalFormat;
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
        long totalStartMs = System.currentTimeMillis();

        try {
            while (completedCount < totalSimulations) {
                Future<SimulationResult> future = completionService.take();
                SimulationResult result = future.get();
                completedCount++;

                Electron electron = result.electron;

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
