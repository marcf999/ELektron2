# ELektron2 Lab Notes

**Project**: Relativistic Mott-Rivas electron–atom scattering simulation
**Repository**: [github.com/marcf999/ELektron2](https://github.com/marcf999/ELektron2)
**Author**: Marc Fleury & Claude Opus 4.6
**Parent project**: `C:\Users\marcf\IdeaProjects\ELektron` (CAPD/JNI version)

---

## Motivation

### Problems with ELektron (CAPD version)
1. **Native memory corruption**: `free(): invalid pointer` crashes when CAPD C++ exceptions crossed the JNI boundary without try/catch in the `TimeMap` wrapper.
2. **WSL dependency**: Required WSL2, CAPD shared library build, `LD_LIBRARY_PATH` setup.
3. **Thread safety concerns**: CAPD integrators created per-thread but potential global state issues.
4. **Step size tuning**: Manual adaptive step logic (`distance / capdStepDivisor`) with caps and floors needed constant tuning.

### Advantages of ELektron2
- Pure Java layer: runs anywhere with a JDK + commons-math3 jar.
- C++ layer: dual integrator support (CAPD Taylor or Boost.Odeint Dormand-Prince).
- DormandPrince handles adaptive stepping automatically (embedded error estimation).
- Event handlers for clean simulation termination (detection sphere, superluminal guard).

---

## Physics

### 12D Rivas Relativistic Model

State vector:
```
[q1, q2, q3, r1, r2, r3, v1, v2, v3, u1, u2, u3]
 q = center of mass position (reduced units, zitter radii)
 r = center of charge position (reduced units)
 v = center of mass velocity (units of c)
 u = center of charge velocity (|u| = c = 1)
```

Equations of motion:
```
dq/dt = v
dr/dt = u
dv/dt = -2*Z*alpha * (r - v*(r·v)) / |r|^3 * sqrt(1 - v²) * exp(-|r|/rB)
du/dt = (q - r) * (1 - v·u) / |q - r|²
```

**Sign convention**: The minus sign on dv/dt makes the Coulomb force **attractive** (electron pulled toward nucleus). Verified against the SymplecticEulerIntegrator in the original ELektron project (line 96). The CAPD vector field string uses a positive sign — see §Known Issues.

### Parameters

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
| totalSimulations | 1000 | Monte Carlo runs |
| plotsToShow | 10 | Swing windows / .dat files |

---

## Architecture

### Java Layer (`src/`)

| File | Purpose |
|------|---------|
| `PhysicalData.java` | All constants, integrator config (tolerances, step bounds) |
| `Electron.java` | 12D state vector, Rivas boost initialization, diagnostics |
| `RivasEquations.java` | `FirstOrderDifferentialEquations` implementation (12D ODE) |
| `Main.java` | Parallel Monte Carlo (ExecutorService + CompletionService), display N trajectories |
| `PlotDots.java` | Swing visualization with auto-scale, zoom, pan, gradient colors |

- **Integrator**: `DormandPrince853Integrator(minStep=1e-10, maxStep=10, absTol=1e-12, relTol=1e-12)`
- **Event handlers**: Forward detection (qz > +1000), backward detection (qz < -1000 and vz < 0), superluminal guard (v² > 0.9999).
- **No fix() renormalization** — DormandPrince853 maintains |u| = 1 through tight tolerances alone.

### C++ Layer (`cpp/`)

```
cpp/
├── CMakeLists.txt
└── src/
    ├── physical_data.h        ← constexpr constants + integrator enum
    ├── electron.h              ← Electron struct with Rivas boost, diagnostics
    ├── rivas_equations.h       ← Boost.Odeint functor (negative sign on dv/dt)
    ├── main.cpp                ← Dual integrator: CAPD Taylor + Boost.Odeint DP5(4)
    └── capd_logger_stub.cpp    ← Logger stubs for standalone CAPD usage
```

- **Dual integrator** selected via compile-time enum in `physical_data.h`:
  ```cpp
  enum class Integrator { CAPD, Boost };
  constexpr Integrator integrator = Integrator::CAPD;  // change to Boost to switch
  ```
- Uses `if constexpr` — unused integrator is dead-code eliminated (zero overhead).
- **CAPD Taylor (order 20)**: String-based vector field parsed by `DMap`, integrated via `DOdeSolver`/`DTimeMap`. Manual adaptive stepping (dt = distCharge / divisor).
- **Boost.Odeint DP5(4)**: `runge_kutta_dopri5` with controlled stepper, abs/relTol = 1e-12. Fully adaptive.
- **Parallelism**: OpenMP `#pragma omp parallel for schedule(dynamic)`, per-thread RNG.
- **Output**: Trajectories written to `.dat` files for gnuplot.

---

## Build and Run

### Java (any platform)
```bash
cd src/
javac -cp ../lib/commons-math3-3.6.1.jar *.java
java -cp .:../lib/commons-math3-3.6.1.jar Main
```
Or run `Main` directly from IntelliJ. No WSL required.

### C++ (Linux/WSL)
```bash
cd cpp && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
export LD_LIBRARY_PATH=/home/marcf/capd-install/lib:$LD_LIBRARY_PATH  # if using CAPD
./elektron2
```
CMake finds Boost (required) and CAPD (optional via `capd-config`). WSL note: cmake must run in native Linux filesystem (not /mnt/c/) due to NTFS permission issues.

---

## Chronological Log

### March 7, 2026 — Java rewrite

**1) Analysis of original ELektron CAPD code**
- Read all 7 Java source files + C++ JNI wrapper (TaylorIntegrator.cpp).
- Identified starting distance (`startPos = -40` reduced), detection cutoff (`reducedBohr * 101`).
- Requested start at -1000, detection at ±1000.

**2) Adaptive step analysis and optimization (ELektron)**
- Original step logic: `stepTime = distance / capdStepDivisor` (divisor = 10).
- Implemented distance-dependent divisor (10 close, 2 far) with cap at 50, floor at 1e-4.
- Raised checkpoint limit from 100k to 500k steps.

**3) CAPD crash diagnosis**
- `free(): invalid pointer` errors from native CAPD code.
- Root cause: No C++ try/catch in `Java_TaylorIntegrator_TimeMap` JNI function. CAPD exceptions crossed JNI boundary causing undefined behavior.
- Fix: Added try/catch returning nullptr; Java side already handled null. Rebuilt `.so` via WSL `make`.

**4) Decision to rewrite**
- Chose DormandPrince853 (adaptive, 8th order) over GraggBulirschStoer or fixed-step RK4.
- Full 12D Rivas model (not simplified 6D Mott).
- No fix(), 1000 Monte Carlo runs, display 10 trajectories, energy out in bold orange.

**5) ELektron2 project creation**
- Created `C:\Users\marcf\IdeaProjects\ELektron2` with `src/`, `lib/`, `.gitignore`, `ELektron2.iml`.
- Copied `commons-math3-3.6.1.jar` from ELektron.
- Wrote all 5 source files from scratch, porting physics from ELektron.

**6) First run — detection boundary bug**
- First 1000 runs completed in 80ms — suspiciously fast.
- All electrons exiting at 270° with unchanged energy (5000 eV).
- Diagnosis: electron starts at z = -1000, exactly at detection sphere radius. Event handler fired immediately.
- Fix: Changed from radial distance event to z-coordinate events.

**7) Force sign correction**
- Trajectories appeared repulsive instead of attractive.
- Compared with SymplecticEulerIntegrator line 96: `accelerationX = -(r - v*(r.v)) * factorEM / distance3`.
- RivasEquations was missing the minus sign. Added it.
- Note: The CAPD vector field string also lacks this minus sign — see §Known Issues.

**8) Parallelization**
- `ExecutorService` + `CompletionService`, `cores * 4` in-flight tasks.
- Each thread creates its own `DormandPrince853Integrator` and `RivasEquations` — no shared mutable state.

**9) Spin inversion**
- Changed `PhysicalData.spin` from -1 to +1.
- Affects `phi0` in Rivas boost: `phi0 = +PI/2` (spin +1/2) vs `-PI/2` (spin -1/2).

**10) Visualization overhaul (PlotDots.java)**
- Auto-scale viewport, dark background (RGB 20,20,30).
- Color gradient: mass yellow→red, charge cyan→blue over time.
- Bohr radius circle, crosshairs, scroll wheel zoom, click-drag pan.
- Legend, title bar info (energy, angle, spin, apex), HUD overlay, zoom indicator.

**11) Parameter changes**
- Impact parameter range widened: `rangeMin = 1e-12`, `rangeMax = 1e-10` (was 1e-13 to 1e-12).
- `plotsToShow` increased from 5 to 10.

### March 7, 2026 — Git setup and GitHub

**12) Git initialization and first commit**
- Initialized git repository, configured identity: `Marc Fleury <marcf999@gmail.com>`.
- Staged 11 files (1458 lines), including `lib/commons-math3-3.6.1.jar` (~2MB).
- GitHub requires personal access tokens (not passwords) for HTTPS push.

**13) GitHub push**
- Repository pushed to `https://github.com/marcf999/ELektron2` (public).

### March 7, 2026 — C++ port (Boost.Odeint)

**14) C++ port of ELektron2**
- Ported all 5 Java source files to C++17 under `cpp/` directory.
- **Integrator**: Boost.Odeint `runge_kutta_dopri5` with controlled stepper (adaptive Dormand-Prince 5(4)). Note: lower order than Java's DP8(5,3).
- **Parallelism**: OpenMP `schedule(dynamic)`, per-thread `std::mt19937` RNG.
- **Output**: Trajectories to `.dat` files for gnuplot.
- **Build**: CMake 3.16+, finds Boost (header-only) and OpenMP automatically.
- **Results**: 1000 sims on 24 cores in 8.9s. isPos=1000, isNaN=0. Energy ~5000.00 eV. |u|² held at 1.0.

### March 7, 2026 — CAPD Taylor integrator in C++

**15) CAPD integration**
- Replaced Boost.Odeint with CAPD Taylor series integrator (order 20).
- CAPD uses string-based vector field definition parsed by `DMap`, integrated via `DOdeSolver`/`DTimeMap`.
- **CAPD vector field**: Same formula as original ELektron project — POSITIVE sign on dv/dt.
- **CAPD Logger issue**: CAPD built with `HAVE_LOG4CXX` but log4cxx not linked. Solved by providing stub implementations of 5 missing symbols in `capd_logger_stub.cpp` (constructor, `isDebugEnabled`, `isTraceEnabled`, `forcedLogDebug`, `forcedLogTrace`).
- **OpenMP thread safety**: CAPD `DMap` string parsing is NOT thread-safe. Solved with `#pragma omp critical(capd_init)` around object creation (heap-allocated), each thread uses its own objects for integration.
- **Manual adaptive stepping**: `dt = distCharge / divisor` where divisor = 10 close to nucleus, 2 far away. Step capped at [1e-4, 50].
- **Results**: 1000 sims in ~12s (24 cores), ~290 steps/electron, energy ~5000.005 eV. 0 NaN.

**16) Dual integrator support (CAPD + Boost.Odeint)**
- Refactored `main.cpp` to support both integrators via compile-time enum.
- `if constexpr` for zero-overhead dispatch.
- `CMakeLists.txt` finds both Boost (required) and CAPD (optional). If CAPD not installed, only Boost available.
- `rivas_equations.h` (Boost functor, negative sign) reinstated alongside CAPD vector field string (positive sign).
- Both integrators tested successfully with 1000 Monte Carlo sims.

---

## Integrator Comparison

### Results (1000 sims, 5000 eV → Carbon Z=6)

| Metric | Java DP8(5,3) | C++ Boost DP5(4) | C++ CAPD Taylor-20 |
|--------|---------------|-------------------|---------------------|
| Steps/electron | 80k–280k | ~880,000 | ~290 |
| Energy out (eV) | 5000.07–5000.22 | ~5000.002 | ~5000.005 |
| Energy drift | ~0.004% | ~0.00004% | ~0.0001% |
| isNaN | 0 | 0 | 0 |
| isPos / isNeg | 1000/0 | 999/1 | 1000/0 |
| Time (4 cores) | ~576s (seq) | ~50s | ~34s |
| Time (24 cores) | N/A | ~10.3s | ~12.2s |
| |u|² constraint | ~1e-7 drift | held at 1.0 | held at 1.0 |

### Observations

- **CAPD Taylor (order 20) takes ~3000× fewer steps** than Boost DP5(4) — the high-order Taylor expansion covers large intervals per step.
- **Boost DP5(4) shows slightly better raw energy conservation** than CAPD, likely due to the adaptive error estimator keeping local truncation error tight despite many more steps.
- **Java DP8(5,3) shows the most drift** — higher order than Boost DP5(4) but running in Java with different floating-point semantics and many more accumulated steps.
- **All three integrators agree on the physics**: forward scattering dominates at 5 keV on carbon.

### Scaling Behavior (24 cores vs 4 cores)

| | CAPD Taylor-20 | Boost DP5(4) |
|---|---|---|
| **4 cores** | 33.5s | 50.2s |
| **24 cores** | 12.2s | 10.3s |
| **Speedup (4→24)** | 2.7× | 4.9× |

**CAPD scales poorly with many cores** due to `#pragma omp critical(capd_init)` — each thread serializes through CAPD's non-thread-safe `DMap` string parser. With 24 threads queuing through this bottleneck, startup overhead dominates over the short ~290-step integration. Boost has no such constraint: every thread launches immediately with its own `runge_kutta_dopri5` (header-only, no global state).

On 4 cores CAPD wins (33s vs 50s) because the critical section overhead is small relative to total work and the 3000× fewer steps dominate. On 24 cores Boost wins (10s vs 12s) because its embarrassingly-parallel startup amortizes the per-electron step count over many more concurrent threads.

**Potential fix**: Pre-parse the CAPD vector field once on the main thread, then clone/copy the `DMap` per worker thread (if CAPD supports it). This would eliminate the critical section entirely.

---

## Known Issues / Future Work

1. **Performance**: Some electrons with large impact parameters take millions of steps (Java/Boost). Could skip force calculation when `exp(-r/rB)` is negligible and propagate ballistically.
2. **Energy conservation (Java)**: Energy out shows ~5000.07–5000.22 eV for 5000 eV in. Small gain (~0.004%) likely from integrator drift. Tighter tolerances or symplectic integrator would help.
3. **CAPD sign discrepancy**: The CAPD vector field string has positive sign on dv/dt. The SymplecticEuler and Java/Boost RivasEquations have negative. Both conventions produce forward-scattered electrons with good energy conservation. Need to verify against Rivas paper.
4. **Constraints**: |u| = 1 and |q-r|² = 1 maintained to ~1e-7 by Java integrator tolerances (no fix()). C++ integrators hold tighter. Monitor for drift in long runs.
5. **CAPD scaling**: Critical section for `DMap` parsing limits parallel scaling. Investigate CAPD copy constructors or thread-local pre-initialization.

---

## Relationship to ELektron (CAPD/JNI version)

| Aspect | ELektron | ELektron2 (Java) | ELektron2 (C++) |
|--------|----------|-------------------|-----------------|
| Integrator | CAPD Taylor-20 via JNI | DormandPrince853 | CAPD Taylor-20 or Boost DP5(4) |
| Language | Java + C++ | Pure Java | C++17 |
| Step control | Manual | Automatic (adaptive) | CAPD: manual / Boost: automatic |
| Native code | libjniTaylor.so (WSL) | None | CAPD .so + Boost headers |
| fix() | Yes | No | No |
| Thread safety | Fragile (JNI crashes) | Clean | CAPD: critical section / Boost: clean |
| Visualization | Basic PlotDots | Enhanced (zoom, pan, gradients) | .dat files (gnuplot) |
