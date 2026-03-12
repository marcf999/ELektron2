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
| atomCount | 30 | carbon atoms along z-axis |
| atomSpacing | 1.42 Å (735.5 reduced) | graphene C-C bond length |
| chainHalfLength | ~10,665 | reduced units |
| startPos | -(chainHalfLength + 4000) | well before first atom |
| detectionDistance | chainHalfLength + 4000 | forward/backward z-cutoff |
| rangeMin | 1e-13 m | impact parameter min |
| rangeMax | 1e-12 m | impact parameter max |
| spin | +1 | spin up (phi0 = +pi/2) |
| Z | 6 | carbon nucleus (per atom) |
| alpha | 0.00730 | fine structure constant |
| reducedBohr | ~275 | Bohr radius in zitter radii |
| totalSimulations | 24 | Monte Carlo runs |
| plotsToShow | 24 | Swing windows / .dat files |

---

## Architecture

### Java Layer (`src/`)

| File | Purpose |
|------|---------|
| `PhysicalData.java` | All constants, integrator config (tolerances, step bounds) |
| `Electron.java` | 12D state vector, Rivas boost initialization |
| `RivasEquations.java` | `FirstOrderDifferentialEquations` implementation (12D ODE) |
| `Main.java` | Parallel Monte Carlo, virtual detector, results file for detected electrons |
| `PlotDots.java` | Swing visualization with auto-scale, zoom, pan, gradient colors |

- **Integrator**: `DormandPrince853Integrator(minStep=1e-10, maxStep=10, absTol=1e-12, relTol=1e-12)`
- **Event handlers**: Forward detection (qz > detectionDistance), backward detection (qz < -detectionDistance and vz < 0), XY-boundary (|qx| or |qy| > 10 Bohr radii), superluminal guard (v² > 0.9999).
- **No fix() renormalization** — DormandPrince853 maintains |u| = 1 through tight tolerances alone.

### C++ Layer (`cpp/`)

```
cpp/
├── CMakeLists.txt
└── src/
    ├── physical_data.h        ← constexpr constants + integrator enum
    ├── electron.h              ← Electron struct with Rivas boost
    ├── rivas_equations.h       ← Boost.Odeint functor (negative sign on dv/dt)
    ├── dp853_integrator.h      ← Self-contained DP8(5,3) matching commons-math3
    ├── main.cpp                ← Triple integrator, virtual detector, detected-only output
    ├── capd_logger_stub.cpp    ← Logger stubs for standalone CAPD usage
    ├── cuda/
    │   └── elektron_cuda.cu    ← NVIDIA GPU-parallel DP853 (CUDA, FP32/FP64)
    └── rocm/
        └── elektron_rocm.cpp   ← AMD GPU-parallel DP853 (HIP, FP32/FP64)
```

- **Dual integrator** selected via compile-time enum in `physical_data.h`:
  ```cpp
  enum class Integrator { Boost, DP853 };
  constexpr Integrator integrator = Integrator::DP853;  // default
  ```
- Uses `if constexpr` — unused integrators are dead-code eliminated (zero overhead).
- **CAPD Taylor**: Removed (March 10, 2026). Was string-based vector field parsed by `DMap` — non-thread-safe, required external `.so` library, poor parallel scaling.
- **Boost.Odeint DP5(4)**: `runge_kutta_dopri5` with controlled stepper, abs/relTol = 1e-12. Fully adaptive.
- **DP853 (self-contained)**: `dp853_integrator.h`, template<int N>, exact Butcher tableau match to Apache commons-math3 DormandPrince853. No external dependencies. Stack arrays only, no dynamic allocation. CUDA-portable. Default integrator.
- **CUDA**: `elektron_cuda.cu`, one thread per electron, `__constant__` memory for Butcher tableau + physics constants. Two CMake targets: FP32 (`--use_fast_math`) and FP64. Shelved on consumer GPUs (1/64 FP64).
- **ROCm/HIP**: `elektron_rocm.cpp`, HIP port of CUDA kernel. Identical physics, `cuda*` API → `hip*` API. Two CMake targets: FP32 and FP64. AMD Instinct GPUs have full FP64 throughput (1/2 of FP32).
- **Parallelism**: OpenMP `#pragma omp parallel for schedule(dynamic)`, per-thread RNG.
- **Termination**: Forward/backward z-detection, XY-boundary (3 Bohr radii), superluminal guard.
- **Virtual detector**: 1m distance, square aperture. Projects final velocity `(vx/vz, vy/vz) * distance` to detector plane. Only detected electrons written to results file and shown in PlotDots.
- **Visualization**: SFML PlotDots with parallel windows (Q/Esc close-all), atom markers on chain. Only detected electrons displayed.
- **Output**: Results `.dat` file with detected electrons only, including `xDet_mm`/`yDet_mm` columns.

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

### CUDA (Linux/WSL)
```bash
cd cpp && mkdir build-cuda && cd build-cuda
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
./elektron2_cuda       # FP32 (fast math, relaxed tolerances)
./elektron2_cuda_fp64  # FP64 (IEEE precise, tight tolerances)
```
Requires CUDA toolkit 12.8+ (`/usr/local/cuda-12.8/`). CMake auto-detects GPU architecture via `CMAKE_CUDA_ARCHITECTURES native`.

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

### March 7, 2026 — SFML PlotDots visualization (C++)

**17) PlotDots for C++**
- Ported Java Swing PlotDots to C++ using SFML 2.5 (graphics, window, system).
- Each trajectory opens its own SFML window in a parallel thread.
- Q or Esc in any window closes all windows.
- Camera capture: z-range based (within atom chain ± 2 atom spacings), decimation factor 200 for Boost.
- Auto-scale viewport, dark background, color gradients (mass yellow→red, charge cyan→blue).
- Atom markers rendered at each nucleus position along the chain.
- Energy output displayed in bold orange 18pt font.

### March 7, 2026 — 30-atom chain and sign fix

**18) 30-atom carbon chain**
- Extended from single atom to 30 carbon atoms along z-axis.
- Graphene C-C bond spacing: 1.42 Å = 735.5 reduced units (zitter radii).
- Chain centered at z=0, half-length ~10,665 reduced units.
- `atomZ[]` array computed via constexpr lambda initializer.
- `startPos` and `detectionDistance` extended to `chainHalfLength + 4000` reduced units.
- RHS sums screened Coulomb force over all 30 atoms: `Σ_k Z*alpha * f(r_k) * exp(-|r_k|/rB) / |r_k|³`.
- CAPD vector field string generated programmatically by `buildRivasVectorField()` (~14KB string for 30 atoms).

**19) CAPD sign fix**
- Discovered the CAPD vector field had the **wrong sign** on dv/dt (positive = repulsive).
- Boost.Odeint and Java always had the correct negative sign (`dv -= emFactor * ...`).
- Fixed CAPD to use negative sign = attractive force.
- With repulsive sign: 0% blowups in all integrators (electrons deflected away from nuclei).
- With correct attractive sign: close encounters cause 1/r³ singularity blowups.

**20) Impact parameter narrowing**
- Narrowed impact parameter range from [1e-12, 1e-10] meters to [1e-13, 1e-12] meters.
- Closer impacts probe deeper into the screened Coulomb potential.
- Reduced `totalSimulations` from 1000 to 24 for manageable run times with 30-atom chain.

### March 7, 2026 — Self-contained DP853 integrator

**21) DP853 integrator (`dp853_integrator.h`)**
- Implemented Dormand-Prince 8(5,3) adaptive integrator as a self-contained C++ header.
- `template<int N>` for compile-time dimensionality (N=12 for Rivas model).
- Exact Butcher tableau match to Apache commons-math3 `DormandPrince853Integrator`:
  - 13 stages (a2..a13), 12 rows of b coefficients
  - Two error estimators: 5th-order and 3rd-order embedded formulas
  - FSAL (First Same As Last) optimization — last stage reused as first stage of next step
- Step control: SAFETY=0.9, MIN_REDUCTION=0.2, MAX_GROWTH=10.0, EXP=-1/8.
- No external dependencies — no Boost, no CAPD, no commons-math3.
- Stack arrays only (no `new`/`malloc`), making it directly CUDA-portable.
- Default tolerances: absTol = relTol = 1e-12, minStep = 1e-10, maxStep = 10.0.
- **Performance**: 3.3× faster than Java DP853 for same physics. ~1.2M steps per electron.
- Set as the default integrator (`PhysicalData::integrator = Integrator::DP853`).

**22) Energy conservation benchmarks (30-atom chain, 5keV)**

*Repulsive sign (pre-fix):*
| Integrator | Blowups | δE per electron | Steps | Time |
|------------|---------|-----------------|-------|------|
| CAPD Taylor 20 | 0/10 | ~1e-7 eV | ~4,200 | ~46s/10 |
| Java DP853 | 0/24 | ~1e-6 eV | ~1.2M | ~57s/24 |
| Boost DP5(4) | 4/10 | 80keV–9.5MeV gains | ~880K | unreliable |

*Attractive sign (post-fix), wide range [1e-12, 1e-10] m:*
| Integrator | Blowups | δE (good electrons) | Backscattered | Time |
|------------|---------|---------------------|---------------|------|
| CAPD Taylor 20 | 4/24 (17%) | 2e-7 to 3e-6 eV | 4 | 230s/24 |
| Java DP853 | 3/24 (13%) | 2e-7 to 3e-6 eV | 2 | 57s/24 |

*Attractive sign, narrow range [1e-13, 1e-12] m:*
| Integrator | Blowups | Forward/Back | δE | Time |
|------------|---------|--------------|-----|------|
| CAPD Taylor 20 | 0/24 | 14/10 | 32keV–4.7MeV gains | 233s/24 |
| Java DP853 | 0/24 | 17/7 | 17keV–3.2MeV gains | 48s/24 |

**Conclusions**:
- Both CAPD and DP853 have similar ~15% blowup rate at wide impact range — the 1/r³ singularity at close approach is a physics limitation, not an integrator defect.
- Narrower range (10× closer) eliminates blowups (0%) but causes massive energy artifacts from deep encounters. This is a model limitation.
- Boost DP5(4) is unreliable (40% blowup even with repulsive sign). Deprecated.

### March 8, 2026 — CUDA port (failed experiment)

**23) CUDA port of DP853 integrator**
- Created `cpp/src/cuda/elektron_cuda.cu` — single self-contained .cu file (~900 lines).
- No external dependencies — all physics constants and Butcher tableau duplicated from C++ headers.
- **Architecture**: One CUDA thread per electron, embarrassingly parallel.
  - All Butcher tableau coefficients (~1.5KB) + physics constants stored in `__constant__` memory for broadcast to all threads.
  - Each thread runs full adaptive DP853 integration independently.
  - Electron initial conditions generated on host (CPU), results copied back after kernel.
- **Dual precision**: `USE_FLOAT` compile macro switches between `typedef float real` and `typedef double real`.
  - FP32 mode: absTol=relTol=1e-6, minStep=1e-8 (relaxed for single precision).
  - FP64 mode: absTol=relTol=1e-12, minStep=1e-10 (matching CPU).
  - Macro wrappers: `REAL_SQRT`, `REAL_EXP`, `REAL_FABS`, `REAL_POW`, `REAL_MAX`, `REAL_MIN`.
- **Optimizations**:
  - Screening cutoff: skip atoms where |Δz| > 5×rB (~1375 reduced units). Reduces inner loop from 30 atoms to ~4-8 nearby atoms.
  - MAX_STEPS safety valve: 2M (FP32) / 5M (FP64) to prevent infinite loops.
  - totalAttempts counter: 20M limit including rejected adaptive steps.
  - Closest approach tracking throttled to every 64 steps.
- **CMake**: Two targets added — `elektron2_cuda` (FP32 + `--use_fast_math -DUSE_FLOAT`) and `elektron2_cuda_fp64` (FP64, `-O2` only).
- **GPU**: NVIDIA GeForce RTX 5070 Laptop GPU, SM 12.0 (Blackwell), 36 SMs, 8151 MB VRAM.
- **CUDA toolkit**: 12.8 installed in WSL at `/usr/local/cuda-12.8/`.

**24) CUDA test results and failure analysis**

*FP64 attempt:*
- Even 1 electron timed out (>120 seconds) with no result.
- Root cause: Consumer GPUs (GeForce) have **1/64 FP64 throughput** vs FP32. The RTX 5070's FP64 rate is ~160 GFLOPS vs ~10 TFLOPS FP32.
- The DP853 integrator is extremely compute-heavy per thread: 13 stages × 30 atoms × exp() + sqrt() per step, repeated ~1.2M+ times.

*FP64 with `--use_fast_math` attempt:*
- Infinite loop — kernel never completed.
- Root cause: `--use_fast_math` reduces precision of `__expf`, `__sqrtf`, `__powf`, etc. The adaptive error estimator saw inflated errors from imprecise transcendentals, rejected steps, shrank step size to minimum, and looped forever trying to achieve 1e-12 tolerance with ~1e-6 precision math.
- **Key lesson**: `--use_fast_math` is fundamentally incompatible with tight-tolerance adaptive integrators.

*FP32 (relaxed tolerances, 1 electron):*
- **Completed**: 2,000,000 steps in 57,214 ms (57 seconds).
- Exit reason: `TIME` — hit MAX_STEPS limit, never reached detection boundary.
- Apex distance: 2069 reduced units (never got close to atoms).
- Energy out: 5000 eV (unchanged — electron didn't interact).
- The electron didn't finish traversing the 30-atom chain in 2M steps.

*Performance comparison:*
| | GPU (FP32) | CPU (FP64, 24 cores) |
|---|---|---|
| Per electron | ~57 seconds | ~0.5 seconds |
| 24 electrons | ~57 seconds (parallel) | ~2 seconds |
| 4096 electrons | ~57 seconds (parallel) | ~85 seconds |
| Precision | 1e-6 | 1e-12 |

**Conclusion**: The CUDA port is a **failed experiment** for this workload on consumer GPUs. The DP853 integrator creates extreme per-thread computational latency (13 RHS evaluations × 30 atoms × transcendentals × millions of steps) that overwhelms GPU parallelism benefits. Consumer GPUs compound this with 1/64 FP64 throughput and the inability to use fast math with adaptive integrators.

The GPU approach could only be viable:
1. On datacenter GPUs (A100/H100/B100) with full FP64 throughput (1/2 of FP32).
2. With massive batches (10K+ electrons) to amortize kernel launch overhead.
3. With a fundamentally different algorithm (e.g., fixed-step RK4 where fast math is safe).

The code is preserved in `cpp/src/cuda/elektron_cuda.cu` for potential future use on datacenter hardware.

### March 8, 2026 — Dead code cleanup

**25) Legacy diagnostics removal (Java + C++)**
- Removed ~15 diagnostic fields from `Electron`: `minimalDistance`, `minimalMassDistance`, `integrationStepTime`, `minStepTime`, `minZelv2`, `maxZelv2`, `minXdot2`, `maxXdot2`, `minR`, `maxR`, `maxGamma`, `isNaN`, `isRenorm`, `isFactorNeg`, `isWellBehaved`.
- Removed methods: `checkIntegrity()`, `getDistanceFromAtomToCharge()`, `getDistanceFromAtomToMass()`, `getuv()`, `getConstraints()`, `getPARAMS()`, `setIntegrationStepTime()`, `isPos()`, `isNeg()`, `is120R()`, `is120L()`.
- Removed `CAMERA_RADIUS` legacy constant (replaced by z-range camera in prior session).
- Made `getGamma()` and `getKineticEnergy()` `const` in C++.
- Removed ANSI color constants, unused `DecimalFormat` import.

**26) State tallying removal (Main.java + main.cpp)**
- Removed all isNaN/isPos/isNeg/is120L/is120R/isRenorm counters and per-electron tallying from the simulation loop.
- Simplified progress logging: just run count + electron exit info + steps + time.
- Removed `debug` flag, `radiusTolerance`, `zdot2Tolerance` from PhysicalData.
- `getEXIT()` simplified: removed Apex distance and Max Gamma fields.
- `getGamma()` no longer sets `isNaN` flag — just returns 1e6 sentinel for v² > 1.
- Exception handlers now print to stderr instead of setting `isNaN`.

**27) Results file cleanup**
- Removed columns: `apexCharge`, `apexMass`, `v2`, `u2`, `|q-r|2`, `minZdot2`, `maxZdot2`, `minXdot2`, `maxXdot2`, `minR`, `maxR`, `isNaN`, `isPos`, `isNeg`.
- Removed `Summary:` header line with tally counts.
- Removed `alpha`, `reducedBohr`, `zitterRadius`, `maxTime` from Java results header.
- Columns now: `idx qx qy qz rx ry rz vx vy vz ux uy uz energyIn_eV energyOut_eV angle_deg steps elapsedMs dxZERO_reduced psi0`.

### March 9, 2026 — Virtual detector and XY-boundary

**28) XY-boundary event handlers**
- New parameter: `xyBoundary = 10.0 * reducedBohr` (~2750 reduced units).
- Electrons that drift too far off-axis (|qx| or |qy| > xyBoundary) are stopped.
- Java: 4 new `EventHandler` instances (±qx, ±qy boundaries).
- C++: Added `std::abs(state[QX]) > xyBoundary || std::abs(state[QY]) > xyBoundary` check in all 3 integrator loops (CAPD, Boost, DP853).
- Purpose: prevents electrons from spiraling indefinitely in the transverse plane after close encounters.

**29) Virtual detector model**
- New parameters in PhysicalData:
  - `detectorDistanceM = 1.0` (1 meter from scattering center).
  - `apertureHalfM`: 0.5e-3 m (1mm × 1mm) in Java, 50e-3 m (100mm × 100mm) in C++.
- Detection logic (C++ `main.cpp`): for forward-going electrons (vz > 0), project velocity to detector plane:
  ```
  xAtDet = (vx / vz) * detectorDistanceM
  yAtDet = (vy / vz) * detectorDistanceM
  ```
  If |xAtDet| < apertureHalfM and |yAtDet| < apertureHalfM → electron is "detected".
- Console output: prints `DETECTED #N (electron i)` with detector coordinates in mm.
- Summary line: `DETECTED: N/total` count.
- Results file: **only writes detected electrons** (not all simulations). Added `xDet_mm` and `yDet_mm` columns.
- PlotDots (C++): only shows trajectories for detected electrons.
- `SimulationResult` struct gains `bool detected = false` field.

**30) Parameter changes for production runs**
- Impact parameter range widened back to `[1e-12, 1e-10]` meters (was `[1e-13, 1e-12]`). Wider range samples both close and distant encounters.
- Simulation count scaled up: Java 240 (was 24), C++ 2400 (was 24).
- `plotsToShow = 0` — visualization disabled by default; only detected electrons are shown.
- All electrons now record camera data (`wantCamera` always true in C++) since only detected ones are displayed.
- Progress logging interval: every 48 completions in C++ (was `progressLogEvery = 100`).
- Detector distance and aperture printed at startup.

### March 9, 2026 — ROCm/HIP port

**31) ROCm/HIP port of DP853 integrator**
- Created `cpp/src/rocm/elektron_rocm.cpp` — mechanical port of `elektron_cuda.cu` to AMD's HIP API.
- All CUDA API calls replaced with HIP equivalents (`cuda*` → `hip*`). Kernel syntax (`<<<grid, block>>>`), device qualifiers (`__constant__`, `__device__`, `__global__`), and math intrinsics are identical in HIP.
- GPU info prints `gcnArchName` (AMD GCN/CDNA architecture) instead of SM version.
- Output file tagged `rocm-dp853` instead of `cuda-dp853`.
- **Motivation**: AMD Instinct GPUs (MI250/MI300) have full FP64 throughput (1/2 of FP32), unlike consumer NVIDIA GPUs (1/64). The DP853 integrator with 1e-12 tolerances requires FP64 — making AMD datacenter GPUs a viable target where NVIDIA consumer GPUs failed.
- **CMake**: Two targets added — `elektron2_rocm` (FP32) and `elektron2_rocm_fp64` (FP64). `find_package(hip QUIET)` detects ROCm installation. Gracefully skips if ROCm not installed.
- **Build**: Requires ROCm toolkit with `hipcc` compiler. CMake links `hip::device`.
  ```bash
  cd cpp && mkdir build-rocm && cd build-rocm
  cmake .. -DCMAKE_BUILD_TYPE=Release
  make elektron2_rocm_fp64
  ./elektron2_rocm_fp64
  ```
- **Not yet tested** — requires AMD GPU hardware with ROCm driver.

---

## Integrator Comparison

### Results (1000 sims, single atom, 5000 eV → Carbon Z=6, repulsive sign)

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

### Results (24 sims, 30-atom chain, 5000 eV, attractive sign, range [1e-13, 1e-12] m)

| Metric | Java DP8(5,3) | C++ DP853 | C++ CAPD Taylor-20 | ROCm FP64 (MI300X) |
|--------|---------------|-----------|---------------------|---------------------|
| Blowups | 0/24 | 0/24 | 0/24 | 15/100K (0.015%) |
| Steps/electron | ~1.2M | ~1.2M | ~4,400–22,000 | 568K |
| δE (eV) | 17keV–3.2MeV | ~1e-6 (good) | 32keV–4.7MeV | ~1e-6 (good) |
| Forward/Back | 17/7 | — | 14/10 | 3565/106 (100K) |
| Time (24 sims) | 48s | ~12s | 233s | ~0.24s (at 100 e/s) |
| Speed vs Java | 1× | 3.3× faster | 4.8× slower | ~16× faster |

### Observations

- **CAPD Taylor (order 20) takes ~3000× fewer steps** than Boost DP5(4) — the high-order Taylor expansion covers large intervals per step.
- **Boost DP5(4) shows slightly better raw energy conservation** than CAPD, likely due to the adaptive error estimator keeping local truncation error tight despite many more steps.
- **Java DP8(5,3) shows the most drift** — higher order than Boost DP5(4) but running in Java with different floating-point semantics and many more accumulated steps.
- **C++ DP853 is the sweet spot**: Same algorithm as Java DP8(5,3) but 3.3× faster due to C++ overhead reduction. No external dependencies. Default integrator.
- **CUDA is not viable** on consumer GPUs — per-thread latency dominates any throughput advantage. **ROCm FP64 on MI300X is the production platform**: full FP64 throughput (1/2 of FP32) makes datacenter AMD GPUs viable where consumer NVIDIA GPUs failed. 100 e/s sustained at 5000 eV.
- **All integrators agree on the physics**: forward scattering dominates at 5 keV on carbon.

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
3. ~~**CAPD sign discrepancy**~~: Fixed in entry §19. All integrators now use negative (attractive) sign on dv/dt.
4. **Constraints**: |u| = 1 and |q-r|² = 1 maintained to ~1e-7 by Java integrator tolerances (no fix()). C++ integrators hold tighter. Monitor for drift in long runs.
5. ~~**CAPD scaling**~~: Resolved by removing CAPD entirely (entry §33).
6. ~~**Detector aperture mismatch**~~: Unified to 100mm × 100mm across all layers (entry §29).
7. ~~**Java detector**~~: Java now implements detector projection with detected-only output (entry §29).
8. **ROCm straggler effect**: Batch completion time dominated by slowest electron. With 10,000 electrons the last few percent can take 2–5× longer than average. Consider adaptive batch sizing or kernel timeout.
9. **GPU occupancy**: DP853 kernel uses ~1.5 KB local memory per thread (k[13][12] doubles), limiting occupancy to ~1 wavefront per CU. Investigate register spilling / LDS usage trade-offs.

---

## Relationship to ELektron (CAPD/JNI version)

| Aspect | ELektron | ELektron2 (Java) | ELektron2 (C++) | ELektron2 (CUDA) | ELektron2 (ROCm) |
|--------|----------|-------------------|-----------------|------------------|------------------|
| Integrator | CAPD Taylor-20 via JNI | DormandPrince853 | Boost / DP853 | DP853 (FP32/FP64) | DP853 (FP32/FP64) |
| Language | Java + C++ | Pure Java | C++17 | CUDA C++ | HIP C++ |
| Step control | Manual | Automatic | CAPD: manual / DP853,Boost: auto | Automatic | Automatic |
| Native code | libjniTaylor.so (WSL) | None | Boost headers (optional) | CUDA toolkit | ROCm toolkit |
| fix() | Yes | No | No | No | No |
| Thread safety | Fragile (JNI crashes) | Clean | Clean (DP853/Boost) | N/A (GPU threads) | N/A (GPU threads) |
| Visualization | Basic PlotDots | Enhanced (zoom, pan) | SFML PlotDots | None (results only) | None (results only) |
| FP64 throughput | N/A | N/A | N/A | 1/64 (consumer) | 1/2 (Instinct) |
| Status | Legacy | Active | Active (default: DP853) | Shelved (consumer GPU too slow) | Active (MI300X tested) |

### March 10, 2026 — Repository housekeeping and cleanup

**32) GitHub default branch fix**
- GitHub repository default branch was `main` (containing only a LICENSE file from repo creation). All code lived on `master`.
- Switched default branch from `main` to `master` via GitHub Settings → Branches so the repo landing page shows the full codebase and LAB-NOTES.md.

**33) Progress output and CAPD removal**
- Added elapsed time to progress output in all three layers (Java, C++, ROCm).
- ROCm: changed from single monolithic kernel launch to batched execution with progress reporting between batches.
- **CAPD removed entirely**: Deleted `capd_logger_stub.cpp`, removed `Integrator::CAPD` enum, `buildRivasVectorField()`, `runCapd()`, CMake CAPD detection, and `-frounding-math` flags. CAPD was non-thread-safe, required external `.so`, and scaled poorly on many cores.
- C++ integrator enum now has only `Boost` and `DP853`.

### March 11, 2026 — ROCm on AMD Instinct MI300X

**34) HotAIsle GPU cloud deployment**
- Deployed to HotAIsle neocloud instance with AMD Instinct MI300X VF (gfx942, 304 CUs, ~192 GB VRAM).
- ROCm 7.2.0, Ubuntu 22.04.
- Build required `-DCMAKE_PREFIX_PATH=/opt/rocm-7.2.0` for CMake to find `hip-config.cmake`.
- Fixed CMakeLists.txt to use CMake HIP language support (`enable_language(HIP)` + `set_source_files_properties(... LANGUAGE HIP)`) — standard `c++` compiler doesn't understand `--offload-arch=gfx942` flag from `hip::device`.

**35) ROCm kernel hang — missing XY boundary check**
- Initial runs hung indefinitely with zero output. Root cause: **the XY boundary check was missing from the ROCm kernel**. The C++ and Java layers stop electrons that scatter beyond `xyBoundary` in the lateral plane, but the GPU kernel had no such check. Electrons trapped in lateral orbits near the atom chain integrated forever.
- Added `EXIT_XY_BOUNDARY` exit code, `d_xyBoundary` constant memory, and lateral escape check: `if (|qx| > xyBoundary || |qy| > xyBoundary)`.
- Also fixed impact parameter range mismatch: ROCm had `[1e-13, 1e-12]`, corrected to `[1e-12, 1e-10]` matching C++/Java.
- Added device-side `printf` checkpoint every 100,000 steps (first 2 threads) for in-kernel progress monitoring.

**36) GPU utilization — batch size tuning**
- First working run: 64 electrons in 213s — but only 1 of 304 CUs was active (`batchSize=64`, `blockSize=64` → `gridSize=1`).
- Increased to `batchSize=totalSimulations`, then tuned to `batchSize=4096` (64 blocks) for balance between GPU fill and progress granularity.
- Full GPU run: **10,000 electrons in 275 seconds** (4.6 minutes).

**37) First GPU vs CPU comparison (10,000 electrons, 5000 eV)**

| | CPU (8 threads, WSL) | GPU (MI300X) |
|---|---|---|
| Time | 26.5 min | 4.6 min |
| Throughput | 6.3 e/s | 36 e/s |
| Detected | 355 (3.5%) | 379 (3.8%) |
| Avg steps | ~1.2M | 809K |
| xyEscape | — | 7,974 (79.7%) |
| Speedup | 1× | **5.8×** |

- Physics matches across platforms: ~3.5–3.8% detection rate at 5 keV on 30-atom carbon chain.
- 79.7% of electrons escape via XY boundary (lateral scatter dominates at these impact parameters).
- GPU per-thread performance is lower than CPU (branchy adaptive integrator with ~1.5 KB register/local memory per thread), but massive parallelism compensates.
- Straggler effect: batch completion time dominated by slowest electron. Batch size of 4096 balances fill vs progress.

**38) XY boundary tightened to 3 Bohr radii**
- Changed `xyBoundary` from `10 * reducedBohr` (~2,745 reduced units) to `3 * reducedBohr` (~823 reduced units) across all three layers (Java, C++, ROCm).
- Tighter boundary terminates laterally-scattered electrons sooner, reducing wasted integration steps.

**39) ROCm throughput optimizations**
- **Block size 64→256**: 4 wavefronts per block instead of 1. Lets the hardware scheduler hide memory latency by switching between wavefronts within a CU.
- **Dynamic batch sizing**: `batchSize = CUs × 256` (= 77,824 on MI300X with 304 CUs) instead of fixed 4096. Ensures all CUs have work.
- **2-stream overlapping**: Two HIP streams alternate batches. While one batch's stragglers finish, the next batch starts on free CUs. Eliminates idle time between batches.
- **Async memcpy**: `hipMemcpyAsync` on the kernel's stream instead of blocking `hipMemcpy`, overlapping D→H transfer with next kernel.
- **Throughput reported**: Summary now prints electrons/sec for easy comparison.

**40) 100K electron benchmark (MI300X, optimized)**

| | 10K (old, §37) | 100K (optimized) |
|---|---|---|
| Block size | 64 | 256 |
| Batch size | 4096 | 77,824 |
| Grid (blocks) | 64 | 304 |
| XY boundary | 10 Bohr | 3 Bohr |
| Throughput | 36 e/s | **96.6 e/s** |
| Avg steps/e | 809K | 568K |
| Detected | 3.8% | 3.6% |
| xyEscape | 79.7% | 96.7% |
| Total time | 275s (10K) | 1036s (100K) |

- **2.7× throughput improvement** (36→96.6 e/s). Gains from: full CU saturation (304/304 vs 64/304), tighter XY boundary (fewer wasted steps), and 4 wavefronts/block for latency hiding.
- Physics consistent: 3.6% detection rate matches previous runs.
- Straggler effect still present: batch 1 (77,824 electrons) took 1035.6s, batch 2 (22,176) completed instantly (overlap with batch 1 stragglers).

**41) Single-point validation run (100K, 5000 eV)**
- 100,000 electrons at 5000 eV: 3,565 detected (3.565%), 99.8 e/s, 15 NaN, 96,671 xyEscape.
- Consistent with §40 benchmarks. Output: `2026-03-11_225243_rocm-dp853_5000eV_100000.dat`.

### March 12, 2026 — Energy scan launch

**42) Energy scan: 750K electrons × 10 energy points**
- Launched full energy scan on HotAisle MI300X: 4991–5009 eV in 2 eV steps, 750,000 electrons per point (7.5M total).
- Fixed `energy_scan.sh` buffering issue: script was capturing all output in a shell variable (`OUTPUT=$(...)`), suppressing live progress. Replaced with `tee` to a temp file so progress prints (every 1000 electrons) stream to console in real time.
- Fixed binary path: script now resolves path relative to its own location (`SCRIPT_DIR/../build/`) instead of assuming CWD.
- Estimated runtime: ~2.1 hours per point, ~21 hours total at ~100 e/s.
- Running in `screen -S scan` to survive SSH disconnect.
  ```bash
  screen -S scan
  cd ~/ELektron2/cpp/scripts && bash energy_scan.sh
  # Detach: Ctrl+A D
  # Reattach: screen -r scan
  # Check log: tail -f ~/ELektron2/cpp/scripts/energy_scan_20260312_*.log
  # Monitor GPU: watch -n 2 rocm-smi
  ```
