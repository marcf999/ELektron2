# 2026-03-07 - ELektron2 Lab Notes - DormandPrince853 Rewrite

## Project
- Repository: `C:\Users\marcf\IdeaProjects\ELektron2`
- Parent project: `C:\Users\marcf\IdeaProjects\ELektron` (CAPD/JNI version)
- Focus: Complete rewrite dropping CAPD/JNI native code in favor of pure Java commons-math3 adaptive integrator.

## Objective
Replace the CAPD Taylor integrator (which required WSL, C++ compilation, JNI bridge, and suffered from `free(): invalid pointer` crashes in multithreaded use) with Apache commons-math3 `DormandPrince853Integrator`. Maintain identical 12D Rivas relativistic physics. Pure Java, no native code.

## Date
- Friday, March 7, 2026

---

## Motivation

### Problems with ELektron (CAPD version)
1. **Native memory corruption**: `free(): invalid pointer` crashes when CAPD C++ exceptions crossed the JNI boundary without try/catch in the `TimeMap` wrapper.
2. **WSL dependency**: Required WSL2, CAPD shared library build, `LD_LIBRARY_PATH` setup.
3. **Thread safety concerns**: CAPD integrators created per-thread but potential global state issues.
4. **Step size tuning**: Manual adaptive step logic (`distance / capdStepDivisor`) with caps and floors needed constant tuning.

### Advantages of ELektron2
- Pure Java: runs anywhere with a JDK + commons-math3 jar.
- DormandPrince853 handles adaptive stepping automatically (8th order, embedded error estimation).
- Event handlers for clean simulation termination (detection sphere, superluminal guard).
- No JNI, no native memory, no `free()` crashes.

---

## Architecture

### Files
| File | Purpose |
|------|---------|
| `PhysicalData.java` | All constants, integrator config (tolerances, step bounds) |
| `Electron.java` | 12D state vector, Rivas boost initialization, diagnostics |
| `RivasEquations.java` | `FirstOrderDifferentialEquations` implementation (12D ODE) |
| `Main.java` | Parallel Monte Carlo (thread pool), timed, display N trajectories |
| `PlotDots.java` | Swing visualization with auto-scale, zoom, pan, gradient colors |

### State Vector (12D)
```
[q1, q2, q3, r1, r2, r3, v1, v2, v3, u1, u2, u3]
 q = center of mass position (reduced)
 r = center of charge position (reduced)
 v = center of mass velocity (units of c)
 u = center of charge velocity (|u| = c = 1)
```

### Equations of Motion (RivasEquations.java)
```
dq/dt = v
dr/dt = u
dv/dt = -2*Z*alpha * (r - v*(r.v)) / |r|^3 * sqrt(1 - v^2) * exp(-|r|/rB)
du/dt = (q - r) * (1 - v.u) / |q - r|^2
```

**Sign convention**: The minus sign on dv/dt makes the Coulomb force **attractive** (electron pulled toward nucleus). This was verified against the SymplecticEulerIntegrator in the original ELektron project (line 96 of that file). The CAPD vector field string was missing this minus sign.

### Integrator
- `DormandPrince853Integrator(minStep=1e-10, maxStep=10, absTol=1e-12, relTol=1e-12)`
- Fully adaptive: no manual step logic needed.

### Event Handlers
1. **Forward detection**: `g = qz - 1000` — stops when electron passes z = +1000 (forward exit).
2. **Backward detection**: `g = qz + 1000` — stops when electron reverses past z = -1000 (backscatter exit). Only fires when `!increasing` (electron heading away).
3. **Superluminal guard**: `g = 0.9999 - v^2` — stops and flags `isNaN` if v approaches c.

### No fix() Renormalization
The original ELektron used `fix()` to renormalize `|u| = 1` after each step. In ELektron2, the DormandPrince853 integrator maintains constraints through tight tolerances (1e-12). No explicit renormalization is applied.

---

## Chronological Log

### 1) Analysis of ELektron CAPD code
- Read all 7 Java source files + C++ JNI wrapper (TaylorIntegrator.cpp).
- Identified starting distance (`startPos = -40` reduced), detection cutoff (`reducedBohr * 101`).
- User requested start at -1000, detection at +/-1000.

### 2) Adaptive step analysis and optimization (ELektron)
- Original step logic: `stepTime = distance / capdStepDivisor` (divisor = 10).
- Implemented distance-dependent divisor (10 close, 2 far) with cap at 50.
- Added floor at 1e-4 to prevent tiny steps near nucleus.
- Raised checkpoint limit from 100k to 500k steps.

### 3) CAPD crash diagnosis
- `free(): invalid pointer` errors from native CAPD code.
- Root cause: No C++ try/catch in `Java_TaylorIntegrator_TimeMap` JNI function. CAPD exceptions crossed JNI boundary causing undefined behavior and memory corruption.
- Fix applied: Added try/catch returning nullptr; Java side already handled null.
- Rebuilt `.so` via WSL `make`.

### 4) Decision to rewrite
- User requested fresh project without CAPD dependency.
- Chose DormandPrince853 (adaptive, 8th order) over GraggBulirschStoer or fixed-step RK4.
- Chose full 12D Rivas model (not simplified 6D Mott).
- User requested: no fix(), 1000 Monte Carlo runs, display 5 (later changed to 10), energy out in standout color.

### 5) ELektron2 project creation
- Created `C:\Users\marcf\IdeaProjects\ELektron2` with `src/`, `lib/`, `.gitignore`, `ELektron2.iml`.
- Copied `commons-math3-3.6.1.jar` from ELektron.
- Wrote all 5 source files from scratch, porting physics from ELektron.

### 6) First run - detection boundary bug
- First 1000 runs completed in 80ms — suspiciously fast.
- All electrons exiting at 270 degrees with unchanged energy (5000eV).
- Diagnosis: electron starts at z = -1000, exactly at the detection sphere radius. Event handler fired immediately.
- Fix: Changed from radial distance event to z-coordinate events (forward at z = +1000, backward at z = -1000).

### 7) Force sign correction
- Trajectories appeared inverted (repulsive instead of attractive).
- Compared with SymplecticEulerIntegrator line 96: `accelerationX = -(r - v*(r.v)) * factorEM / distance3`.
- RivasEquations was missing the minus sign. Added it.
- Note: The CAPD vector field string also lacks this minus sign — unclear if that was intentional or a bug.

### 8) Parallelization
- Converted sequential loop to `ExecutorService` + `CompletionService` pattern (same as original ELektron).
- `cores * 4` in-flight tasks, results collected as completed.
- Each thread creates its own `DormandPrince853Integrator` and `RivasEquations` — no shared mutable state.

### 9) Spin inversion
- Changed `PhysicalData.spin` from -1 to +1.
- This affects `phi0` in the Rivas boost: `phi0 = +PI/2` (spin +1/2) vs `phi0 = -PI/2` (spin -1/2).

### 10) Visualization overhaul (PlotDots.java)
Rewrote PlotDots with:
- **Auto-scale**: viewport fits trajectory bounding box automatically.
- **Dark background**: navy/black (RGB 20,20,30).
- **Color gradient**: mass yellow->red, charge cyan->blue over time.
- **Bohr radius circle**: dashed ring at 5.3e-11m.
- **Crosshairs**: faint axis lines through nucleus.
- **Scroll wheel zoom**: MouseWheelListener scales viewport.
- **Click-drag pan**: MouseMotionListener for panning.
- **Legend**: bottom-right box with color key.
- **Title bar info**: energy, angle, spin, apex distance.
- **HUD overlay**: semi-transparent background for text readability.
- **Zoom indicator**: bottom-left shows current zoom level.

### 11) Parameter changes by user
- Impact parameter range widened: `rangeMin = 1e-12`, `rangeMax = 1e-10` (was 1e-13 to 1e-12).
- `plotsToShow` increased from 5 to 10.

---

## Current Parameters (PhysicalData.java)

| Parameter | Value | Notes |
|-----------|-------|-------|
| startEnergy | 5000 eV | 5 keV electron |
| startPos | -1000 | zitter radii from nucleus |
| detectionDistance | 1000 | forward/backward z-cutoff |
| rangeMin | 1e-12 m | impact parameter min |
| rangeMax | 1e-10 m | impact parameter max |
| spin | +1 | spin up (phi0 = +pi/2) |
| Z | 6 | carbon nucleus |
| alpha | 0.00730 | fine structure constant |
| reducedBohr | ~275 | Bohr radius in zitter radii |
| relTol/absTol | 1e-12 | DormandPrince853 tolerances |
| totalSimulations | 1000 | Monte Carlo runs |
| plotsToShow | 10 | Swing windows displayed |

---

## Build and Run

### Dependencies
- Java JDK (tested with OpenJDK 25)
- `commons-math3-3.6.1.jar` in `lib/`

### Compile (from src/)
```bash
javac -cp ../lib/commons-math3-3.6.1.jar *.java
```

### Run (from src/)
```bash
java -cp .:../lib/commons-math3-3.6.1.jar Main
```

Or run `Main` directly from IntelliJ (module classpath includes the jar via `ELektron2.iml`).

### No WSL required
Pure Java — runs on Windows, macOS, Linux without native libraries.

---

## Performance Notes
- 1000 simulations with corrected force: ~576 seconds (sequential), expect ~N/cores with parallel.
- Most time spent on electrons with large impact parameters (long trajectories, many steps).
- DormandPrince853 typically takes 80k-280k steps per electron (vs 2-3 steps when detection was broken).
- Occasional outlier: 1.7M steps for very large impact parameter electrons.

---

## Known Issues / Future Work
1. **Performance**: Some electrons with large impact parameters take millions of steps. Could skip force calculation when `exp(-r/rB)` is negligible and propagate ballistically.
2. **Energy conservation**: Energy out shows ~5000.07-5000.22 eV for 5000 eV in. Small gain (~0.004%) likely from integrator drift over many steps. Tighter tolerances or symplectic integrator would help.
3. **CAPD sign discrepancy**: The CAPD vector field string in ELektron has positive sign on dv/dt. The SymplecticEuler has negative. ELektron2 uses negative (attractive). Need to verify which is correct against Rivas paper.
4. **Constraints**: |u| = 1 and |q-r|^2 = 1 maintained to ~1e-7 by integrator tolerances alone (no fix()). Monitor for drift in long runs.

---

## Files Updated During This Session

### New files (ELektron2)
- `src/PhysicalData.java` — constants and config
- `src/Electron.java` — 12D state, Rivas boost, diagnostics
- `src/RivasEquations.java` — 12D ODE implementation
- `src/Main.java` — parallel Monte Carlo runner
- `src/PlotDots.java` — enhanced Swing visualization
- `ELektron2.iml` — IntelliJ module file
- `.gitignore`

### Modified files (ELektron, earlier in session)
- `src/PhysicalData.java` — startPos -40 -> -1000
- `src/Main.java` — detection cutoff, adaptive step with cap/floor, checkpoint 500k
- `src/TaylorIntegrator.cpp` — added try/catch in TimeMap JNI wrapper
- `src/makefile` — rebuilt .so

---

## Version Control Setup

### 12) Git initialization and first commit
- Initialized git repository in `C:\Users\marcf\IdeaProjects\ELektron2`.
- Configured git identity: `Marc Fleury <marcf999@gmail.com>`.
- Staged all project files (11 files, 1458 lines):
  - `.gitignore`
  - `ELektron2.iml`
  - `LAB NOTES/` (3 files — original ELektron notes + this session's notes)
  - `lib/commons-math3-3.6.1.jar` (~2MB, included so repo is self-contained)
  - `src/` (5 Java source files)
- Initial commit message:
  ```
  Initial commit: ELektron2 - Relativistic electron-atom scattering
  Pure Java rewrite of ELektron (CAPD/JNI). 12D Rivas model with
  DormandPrince853 adaptive integrator (commons-math3). Monte Carlo
  electron scattering off carbon (Z=6) with screened Coulomb potential.
  Co-Authored-By: Marc Fleury and Claude Opus 4.6
  ```
- GitHub CLI (`gh`) not installed on this machine.
- To publish: create repo at github.com, then `git remote add origin <url>` + `git push -u origin master`.
- GitHub requires personal access tokens (not passwords) for HTTPS push authentication.

---

## C++ Port (Boost.Odeint + OpenMP)

### 13) C++ port of ELektron2
- Ported all 5 Java source files to C++17 under `cpp/` directory.
- **Integrator**: Boost.Odeint `runge_kutta_dopri5` with controlled stepper (adaptive Dormand-Prince 5(4)).
  - Note: Boost has DP5(4), not DP8(5,3) like commons-math3. Same family, lower order.
- **Parallelism**: OpenMP `#pragma omp parallel for schedule(dynamic)` — replaces Java's ExecutorService/CompletionService pattern. Each thread gets its own `std::mt19937` RNG.
- **Visualization**: No GUI library yet. Trajectories written to `.dat` files for gnuplot or future SFML integration.
- **Build system**: CMake 3.16+, finds Boost (header-only) and OpenMP automatically.

### C++ project structure
```
cpp/
├── CMakeLists.txt
└── src/
    ├── physical_data.h      ← PhysicalData namespace (constexpr constants)
    ├── electron.h            ← Electron struct with Rivas boost, diagnostics
    ├── rivas_equations.h     ← RivasEquations functor for Boost.Odeint
    └── main.cpp              ← OpenMP parallel Monte Carlo + .dat file output
```

### Build instructions (Linux/WSL/Ubuntu)
```bash
cd cpp && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
./elektron2
```

### Test results
- 1000 simulations on 24 cores: **8.9 seconds**
- All 1000 electrons forward-scattered (isPos=1000, isNaN=0)
- Energy conservation: ~5000.00 eV out for 5000 eV in
- Constraint |u|²: held at 1.0 throughout
- WSL note: cmake must run in native Linux filesystem (not /mnt/c/) due to NTFS permission issues with cmake cache files.

### 14) GitHub push
- Repository pushed to `https://github.com/marcf999/ELektron2` (public).
- 3 commits: initial Java, lab notes update, C++ port.

---

## Relationship to ELektron (CAPD version)

ELektron2 is a **clean rewrite** of ELektron with the same physics but different numerics:

| Aspect | ELektron | ELektron2 |
|--------|----------|-----------|
| Integrator | CAPD Taylor (order 20) via JNI | DormandPrince853 (commons-math3) |
| Language | Java + C++ | Pure Java |
| Step control | Manual (distance/divisor + cap/floor) | Automatic (adaptive embedded error) |
| Native code | libjniTaylor.so (WSL required) | None |
| fix() | Yes (renormalize u each step) | No (tolerances handle it) |
| Thread safety | Fragile (JNI crashes) | Clean (no shared state) |
| Visualization | Basic PlotDots | Enhanced (zoom, pan, gradients, auto-scale) |
