# 2026-03-05 - ELektron Lab Notes - WSL2 + CAPD + JNI Integration

## Project
- Repository: `C:\Users\marcf\IdeaProjects\ELektron`
- Focus area: `src/TaylorIntegrator.java`, `src/TaylorIntegrator.cpp`, JNI build flow, CAPD integration.

## Objective
Establish a reproducible Linux-native (WSL2/Ubuntu) build and runtime pipeline for CAPD-backed JNI integration so `TaylorIntegrator` compiles and runs from the project successfully.

## High-Level Outcome
- WSL2 installed and usable.
- CAPD installation verified at `/home/marcf/capd-install`.
- JNI makefile migrated from macOS-specific settings to WSL2/Linux settings.
- JNI shared library builds successfully: `libjniTaylor.so`.
- Java runtime successfully loads JNI and executes CAPD computations.
- Native-access warning addressed with `--enable-native-access=ALL-UNNAMED`.
- Build/run instructions updated.

## Date
- Thursday, March 5, 2026

---

## Environment and Tooling

### Windows Host
- OS shell used in agent session: PowerShell.
- Initial state: `java` and `javac` were not on global Windows `PATH`.
- IntelliJ local JDK (Windows side) discovered in:
  - `C:\Users\marcf\.jdks\openjdk-25.0.2`

### WSL/Ubuntu
- WSL eventually installed successfully.
- CAPD installation path verified from WSL:
  - `/home/marcf/capd-install`
  - Contains `bin`, `include`, `lib`.
- JDK in WSL resolved via `javac`:
  - `/usr/lib/jvm/java-25-openjdk-amd64/bin/javac`
- Effective WSL JAVA_HOME target:
  - `/usr/lib/jvm/java-25-openjdk-amd64`

---

## Chronological Log (Lab Style)

### 1) Initial codebase analysis and parallelism review
- Reviewed simulation architecture (`Main`, `Electron`, `SymplecticEulerIntegrator`).
- Confirmed simulation-level parallelism via fixed thread pool + completion service.
- Identified correctness/performance issues and implemented requested fixes:
  - NaN stop condition bug.
  - Numerical guards around divisions and non-finite values.
  - Safe normalization in `fix()`.
  - Bounded in-flight task submission.
  - Reduced allocation pressure in integrator hot path.
  - Camera/history memory-control improvements.
- Multiple regressions were then corrected where introduced (notably `Main.java` brace/logic repair and method/API consistency).

### 2) Java toolchain availability troubleshooting (Windows shell)
- `javac`/`java` initially failed as not found from PowerShell (`CommandNotFoundException`).
- Located IntelliJ-configured JDK and confirmed direct binary invocation works.
- Verified:
  - `javac 25.0.2`
  - `java 25.0.2`

### 3) Compile and run validation in project
- Compile initially failed due missing dependency (`commons-math3`) when `lib` was empty.
- After jar update, compile surfaced syntax errors in `Main.java`; block structure repaired.
- Full compile and run restored.
- Added configurable simulation count and later switched back to `PhysicalData.totalSimulations` per request.
- Added final lineout:
  - `SYSTEM_LINEOUT | iterations=... | elapsedMs=...`
- Set simulation iterations to `10000` in `PhysicalData`.

### 4) WSL2 setup planning and execution
- User requested CAPD + JNI plan; recommended WSL2 over MSYS2 for this stack.
- Initial WSL install attempt failed (`wsl` not recognized).
- Guidance provided to enable features/reboot; user confirmed WSL installed.

### 5) CAPD install verification and build configuration
- CAPD path verified in WSL:
  - `/home/marcf/capd-install`
- CAPD compile/link flags provided by user from `capd-config --cflags --libs`:
  - `-I/home/marcf/capd-install/include ... -L/home/marcf/capd-install/lib -lcapd -lfilib`
- Existing makefile identified as macOS-specific:
  - hardcoded mac JAVA_HOME
  - `darwin` JNI includes
  - `.jnilib` target

### 6) Makefile migration to WSL/Linux JNI
- Updated `src/makefile` to:
  - `JAVA_HOME ?= /usr/lib/jvm/java-25-openjdk-amd64`
  - `CAPD_CONFIG ?= /home/marcf/capd-install/bin/capd-config`
  - JNI include dirs: `include` + `include/linux`
  - output target: `libjniTaylor.so`
  - `run` target with `LD_LIBRARY_PATH` and JNI library path.
- Dry-run verification (`make -n`) confirmed expected compile command generation.

### 7) Native linker failure analysis and fix
- Encountered linker error while building JNI `.so`:
  - relocation errors from `/home/marcf/capd-install/lib/libcapd.a`
  - message indicated non-PIC static objects.
- Diagnosis: static CAPD archive not suitable for linking shared JNI library.
- Fix applied: rebuild CAPD with shared/PIC settings (user completed).
- Verification after rebuild:
  - `libcapd.so` present at `/home/marcf/capd-install/lib/libcapd.so`.

### 8) Final successful JNI build and run
- Rebuilt from WSL project source directory.
- Successful JNI build command path produced `libjniTaylor.so`.
- Runtime executed successfully with CAPD-backed integration output.
- Demonstration output included:
  - Time map,
  - derivative matrix,
  - Poincare map,
  - periodic orbit sequence,
  - return consistency checks.

### 9) JDK native-access warning mitigation
- Runtime warning observed (JDK 25 restricted native access warning).
- Standard mitigation adopted:
  - `--enable-native-access=ALL-UNNAMED`

### 10) Build instruction updates
- Updated `src/compile-run-instructions.txt` run command to include:
  - `LD_LIBRARY_PATH=/home/marcf/capd-install/lib:$LD_LIBRARY_PATH`
  - `--enable-native-access=ALL-UNNAMED`
  - `-Djava.library.path=.`
- Updated makefile `run` target accordingly.

---

## Commands Used (Canonical WSL Flow)

From WSL:

```bash
cd /mnt/c/Users/marcf/IdeaProjects/ELektron/src
javac TaylorIntegrator.java
javac -h . TaylorIntegrator.java
make
LD_LIBRARY_PATH=/home/marcf/capd-install/lib:$LD_LIBRARY_PATH \
java --enable-native-access=ALL-UNNAMED -Djava.library.path=. TaylorIntegrator
```

Optional convenience:

```bash
make run
```

---

## Files Updated During This Session

### Core project files
- `src/makefile`
  - Migrated to WSL/Linux JNI build settings.
- `src/compile-run-instructions.txt`
  - Updated runtime command for JNI + CAPD + native access.

### Earlier simulation/runtime edits (same working session)
- `src/Main.java`
  - simulation control and output changes, including system lineout.
- `src/PhysicalData.java`
  - simulation count setting via `totalSimulations`.
- `src/Electron.java`
  - safety/behavior fixes in selected methods.
- `src/SymplecticEulerIntegrator.java`
  - numerical guard hardening.
- `src/PlotDots.java`
  - camera iteration compatibility updates.

---

## Known Working Paths

### WSL paths
- Project source: `/mnt/c/Users/marcf/IdeaProjects/ELektron/src`
- CAPD install: `/home/marcf/capd-install`
- WSL Java home: `/usr/lib/jvm/java-25-openjdk-amd64`

### Windows paths
- Project root: `C:\Users\marcf\IdeaProjects\ELektron`
- Windows-side JDK used earlier for non-WSL compile checks:
  - `C:\Users\marcf\.jdks\openjdk-25.0.2`

---

## Recommended Next Steps

1. Keep all CAPD/JNI build+run actions in WSL to avoid cross-OS library mismatches.
2. Optionally add a dedicated `README-WSL-CAPD.md` with only the stable final commands.
3. Optionally pin Java version across Windows and WSL (both at JDK 25 currently) for consistency.
4. Optionally create IntelliJ run configuration that shells out through WSL and uses the exact `make run` path.

---

## Final Status (End of Session)
- CAPD JNI integration: WORKING in WSL.
- `TaylorIntegrator` build: PASS.
- `TaylorIntegrator` run: PASS.
- Runtime warnings mitigated in documented command.
