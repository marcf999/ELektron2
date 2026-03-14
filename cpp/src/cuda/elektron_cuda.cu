// ============================================================================
// ELektron2 CUDA — GPU-parallel electron scattering simulation
//
// One CUDA thread per electron, each runs the full DP853 adaptive integration.
// All Butcher tableau + physics constants in __constant__ memory (~1.5KB).
// No shared memory needed — embarrassingly parallel.
//
// Build: nvcc -O2 -arch=native -o elektron2_cuda elektron_cuda.cu
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <ctime>
#include <chrono>
#include <random>
#include <vector>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>

// ============================================================================
// CUDA error checking
// ============================================================================
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// ============================================================================
// Physics constants (mirrored from physical_data.h)
// These are duplicated here to keep the CUDA build self-contained.
// ============================================================================
namespace PhysConst {
    constexpr int atomCount = 30;
    constexpr double zitterRadius = 1.93079634e-13;
    constexpr double atomSpacingMeters = 1.42e-10;
    constexpr double atomSpacing = atomSpacingMeters / zitterRadius;
    constexpr double chainHalfLength = (atomCount - 1) / 2.0 * atomSpacing;

    constexpr double startEnergy = 5000.0;        // eV
    constexpr double startPos = -(chainHalfLength + 4000.0);
    constexpr double detectionDistance = chainHalfLength + 4000.0;
    constexpr double maxTime = 1e6;

    constexpr double rangeMin = 1e-13;
    constexpr double rangeMax = 1e-12;

    // Spin axis: set at runtime via CLI (default: +z)
    inline double spinTheta0 = 0.0;
    inline double spinPhi0   = 0.0;
    inline std::string spinLabel = "+z";

    inline bool spinRandom = false;  // when true, each electron gets a random spin axis

    inline void setSpinAxis(const std::string& axis) {
        if (axis == "+z" || axis == "z")       { spinTheta0 = 0.0;        spinPhi0 = 0.0;           spinLabel = "+z"; }
        else if (axis == "-z")                 { spinTheta0 = M_PI;       spinPhi0 = 0.0;           spinLabel = "-z"; }
        else if (axis == "+x" || axis == "x")  { spinTheta0 = M_PI/2.0;  spinPhi0 = 0.0;           spinLabel = "+x"; }
        else if (axis == "-x")                 { spinTheta0 = M_PI/2.0;  spinPhi0 = M_PI;          spinLabel = "-x"; }
        else if (axis == "+y" || axis == "y")  { spinTheta0 = M_PI/2.0;  spinPhi0 = M_PI/2.0;     spinLabel = "+y"; }
        else if (axis == "-y")                 { spinTheta0 = M_PI/2.0;  spinPhi0 = 3.0*M_PI/2.0;  spinLabel = "-y"; }
        else if (axis == "random" || axis == "rand") { spinRandom = true; spinLabel = "random"; }
        else {
            fprintf(stderr, "Unknown spin axis: '%s'. Use: +x,-x,+y,-y,+z,-z,random\n", axis.c_str());
            exit(1);
        }
    }

    constexpr double carbonProtons = 6.0;
    constexpr double alpha = 0.007299270072992700;
    constexpr double bohrRadius = 5.3e-11;
    constexpr double m0c2 = 5.11e+5;

    constexpr double dp853AbsTol = 1e-12;
    constexpr double dp853RelTol = 1e-12;
    constexpr double dp853MinStep = 1e-10;
    constexpr double dp853MaxStep = 10.0;

    constexpr int defaultSimulations = 1024;
    constexpr int progressLogEvery = 100;

    // Computed at startup
    inline double reducedBohr;
    inline double atomZ[atomCount];

    void init() {
        reducedBohr = bohrRadius / zitterRadius;
        for (int i = 0; i < atomCount; i++) {
            atomZ[i] = (i - (atomCount - 1) / 2.0) * atomSpacing;
        }
    }
}

// ============================================================================
// Butcher tableau for DP853 (computed on host, copied to __constant__)
// ============================================================================
namespace ButcherHost {
    static const double S6 = std::sqrt(6.0);

    static const double C[12] = {
        (12.0 - 2.0*S6) / 135.0,
        (6.0 - S6) / 45.0,
        (6.0 - S6) / 30.0,
        (6.0 + S6) / 30.0,
        1.0 / 3.0,
        1.0 / 4.0,
        4.0 / 13.0,
        127.0 / 195.0,
        3.0 / 5.0,
        6.0 / 7.0,
        1.0,
        1.0
    };

    static const double A[12][12] = {
        {(12.0 - 2.0*S6) / 135.0},
        {(6.0 - S6) / 180.0, (6.0 - S6) / 60.0},
        {(6.0 - S6) / 120.0, 0.0, (6.0 - S6) / 40.0},
        {(462.0 + 107.0*S6) / 3000.0, 0.0,
         (-402.0 - 197.0*S6) / 1000.0, (168.0 + 73.0*S6) / 375.0},
        {1.0/27.0, 0.0, 0.0, (16.0 + S6) / 108.0, (16.0 - S6) / 108.0},
        {19.0/512.0, 0.0, 0.0,
         (118.0 + 23.0*S6) / 1024.0, (118.0 - 23.0*S6) / 1024.0,
         -9.0/512.0},
        {13772.0/371293.0, 0.0, 0.0,
         (51544.0 + 4784.0*S6) / 371293.0, (51544.0 - 4784.0*S6) / 371293.0,
         -5688.0/371293.0, 3072.0/371293.0},
        {58656157643.0/93983540625.0, 0.0, 0.0,
         (-1324889724104.0 - 318801444819.0*S6) / 626556937500.0,
         (-1324889724104.0 + 318801444819.0*S6) / 626556937500.0,
         96044563816.0/3480871875.0, 5682451879168.0/281950621875.0,
         -165125654.0/3796875.0},
        {8909899.0/18653125.0, 0.0, 0.0,
         (-4521408.0 - 1137963.0*S6) / 2937500.0,
         (-4521408.0 + 1137963.0*S6) / 2937500.0,
         96663078.0/4553125.0, 2107245056.0/137915625.0,
         -4913652016.0/147609375.0, -78894270.0/3880452869.0},
        {-20401265806.0/21769653311.0, 0.0, 0.0,
         (354216.0 + 94326.0*S6) / 112847.0,
         (354216.0 - 94326.0*S6) / 112847.0,
         -43306765128.0/5313852383.0, -20866708358144.0/1126708119789.0,
         14886003438020.0/654632330667.0,
         35290686222309375.0/14152473387134411.0,
         -1477884375.0/485066827.0},
        {39815761.0/17514443.0, 0.0, 0.0,
         (-3457480.0 - 960905.0*S6) / 551636.0,
         (-3457480.0 + 960905.0*S6) / 551636.0,
         -844554132.0/47026969.0, 8444996352.0/302158619.0,
         -2509602342.0/877790785.0,
         -28388795297996250.0/3199510091356783.0,
         226716250.0/18341897.0, 1371316744.0/2131383595.0},
        {104257.0/1920240.0, 0.0, 0.0, 0.0, 0.0,
         3399327.0/763840.0, 66578432.0/35198415.0,
         -1674902723.0/288716400.0, 54980371265625.0/176692375811392.0,
         -734375.0/4826304.0, 171414593.0/851261400.0, 137909.0/3084480.0}
    };

    static const double B[13] = {
        104257.0/1920240.0, 0.0, 0.0, 0.0, 0.0,
        3399327.0/763840.0, 66578432.0/35198415.0,
        -1674902723.0/288716400.0, 54980371265625.0/176692375811392.0,
        -734375.0/4826304.0, 171414593.0/851261400.0, 137909.0/3084480.0,
        0.0
    };

    static const double E1[12] = {
        116092271.0/8848465920.0,
        0.0, 0.0, 0.0, 0.0,
        -1871647.0/1527680.0,
        -69799717.0/140793660.0,
        1230164450203.0/739113984000.0,
        -1980813971228885.0/5654156025964544.0,
        464500805.0/1389975552.0,
        1606764981773.0/19613062656000.0,
        -137909.0/6168960.0
    };

    static const double E2[12] = {
        -364463.0/1920240.0,
        0.0, 0.0, 0.0, 0.0,
        3399327.0/763840.0,
        66578432.0/35198415.0,
        -1674902723.0/288716400.0,
        -74684743568175.0/176692375811392.0,
        -734375.0/4826304.0,
        171414593.0/851261400.0,
        69869.0/3084480.0
    };
}

// ============================================================================
// Precision selection: compile with -DUSE_FLOAT for FP32 (64x faster on consumer GPUs)
// FP64 is IEEE-precise but consumer GPUs (GeForce) have 1/64 FP64 throughput.
// FP32 with relaxed tolerances gives physically meaningful results much faster.
// ============================================================================
#ifdef USE_FLOAT
typedef float  real;
#define REAL_SQRT  sqrtf
#define REAL_EXP   expf
#define REAL_FABS  fabsf
#define REAL_FMIN  fminf
#define REAL_FMAX  fmaxf
#define REAL_POW   powf
#else
typedef double real;
#define REAL_SQRT  sqrt
#define REAL_EXP   exp
#define REAL_FABS  fabs
#define REAL_FMIN  fmin
#define REAL_FMAX  fmax
#define REAL_POW   pow
#endif

// ============================================================================
// CUDA __constant__ memory — broadcast to all threads, cached
// ============================================================================

// Physics
__constant__ real d_atomZ[30];
__constant__ real d_Z;
__constant__ real d_alpha;
__constant__ real d_rB;
__constant__ real d_detectionDistance;
__constant__ real d_maxTime;
__constant__ real d_m0c2;

// Butcher tableau
__constant__ real d_C[12];
__constant__ real d_A[12][12];
__constant__ real d_B[13];
__constant__ real d_E1[12];
__constant__ real d_E2[12];

// Integrator parameters
__constant__ real d_absTol;
__constant__ real d_relTol;
__constant__ real d_minStep;
__constant__ real d_maxStep;

// ============================================================================
// Copy all constants from host to device __constant__ memory
// ============================================================================
void initDeviceConstants() {
    using namespace PhysConst;

    // Convert all constants to the working precision (float or double)
    real h_atomZ[30];
    for (int i = 0; i < 30; i++) h_atomZ[i] = (real)atomZ[i];
    CUDA_CHECK(cudaMemcpyToSymbol(d_atomZ, h_atomZ, sizeof(h_atomZ)));

    real val;
    val = (real)carbonProtons;  CUDA_CHECK(cudaMemcpyToSymbol(d_Z, &val, sizeof(real)));
    val = (real)alpha;          CUDA_CHECK(cudaMemcpyToSymbol(d_alpha, &val, sizeof(real)));
    val = (real)reducedBohr;    CUDA_CHECK(cudaMemcpyToSymbol(d_rB, &val, sizeof(real)));
    val = (real)detectionDistance; CUDA_CHECK(cudaMemcpyToSymbol(d_detectionDistance, &val, sizeof(real)));
    val = (real)maxTime;        CUDA_CHECK(cudaMemcpyToSymbol(d_maxTime, &val, sizeof(real)));
    val = (real)m0c2;           CUDA_CHECK(cudaMemcpyToSymbol(d_m0c2, &val, sizeof(real)));

    // Butcher tableau
    real h_C[12], h_A[12][12], h_B[13], h_E1[12], h_E2[12];
    for (int i = 0; i < 12; i++) {
        h_C[i] = (real)ButcherHost::C[i];
        h_E1[i] = (real)ButcherHost::E1[i];
        h_E2[i] = (real)ButcherHost::E2[i];
        for (int j = 0; j < 12; j++) h_A[i][j] = (real)ButcherHost::A[i][j];
    }
    for (int i = 0; i < 13; i++) h_B[i] = (real)ButcherHost::B[i];
    CUDA_CHECK(cudaMemcpyToSymbol(d_C, h_C, sizeof(h_C)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_A, h_A, sizeof(h_A)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_B, h_B, sizeof(h_B)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_E1, h_E1, sizeof(h_E1)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_E2, h_E2, sizeof(h_E2)));

    // Integrator tolerances
#ifdef USE_FLOAT
    // Float mode: relax tolerances (7 digits precision)
    val = (real)1e-6;  CUDA_CHECK(cudaMemcpyToSymbol(d_absTol, &val, sizeof(real)));
    val = (real)1e-6;  CUDA_CHECK(cudaMemcpyToSymbol(d_relTol, &val, sizeof(real)));
    val = (real)1e-8;  CUDA_CHECK(cudaMemcpyToSymbol(d_minStep, &val, sizeof(real)));
#else
    val = (real)dp853AbsTol; CUDA_CHECK(cudaMemcpyToSymbol(d_absTol, &val, sizeof(real)));
    val = (real)dp853RelTol; CUDA_CHECK(cudaMemcpyToSymbol(d_relTol, &val, sizeof(real)));
    val = (real)dp853MinStep; CUDA_CHECK(cudaMemcpyToSymbol(d_minStep, &val, sizeof(real)));
#endif
    val = (real)dp853MaxStep; CUDA_CHECK(cudaMemcpyToSymbol(d_maxStep, &val, sizeof(real)));
}

// ============================================================================
// Device: Rivas equations right-hand side
// ============================================================================
// Screening cutoff: skip atoms where exp(-r/rB) < threshold
// At 5*rB: exp(-5)=0.007, force contribution ~1e-12 — negligible
#define SCREEN_CUTOFF_RB  5.0

#define ATOM_COUNT 30

__device__ void rivas_rhs(real /*t*/, const real* __restrict__ y,
                          real* __restrict__ dydt) {
    real q1 = y[0], q2 = y[1], q3 = y[2];
    real r1 = y[3], r2 = y[4], r3 = y[5];
    real v1 = y[6], v2 = y[7], v3 = y[8];
    real u1 = y[9], u2 = y[10], u3 = y[11];

    // dq/dt = v
    dydt[0] = v1;  dydt[1] = v2;  dydt[2] = v3;
    // dr/dt = u
    dydt[3] = u1;  dydt[4] = u2;  dydt[5] = u3;

    // dv/dt: screened Coulomb force summed over all atoms
    real v2sq = v1*v1 + v2*v2 + v3*v3;
    real sqrtFactor = REAL_SQRT(REAL_FMAX((real)1.0 - v2sq, (real)0.0));
    real twoZAlpha = (real)2.0 * d_Z * d_alpha;

    real dv1 = (real)0.0, dv2 = (real)0.0, dv3 = (real)0.0;

    // Screening cutoff: skip atoms where |z-distance| > cutoff
    // (full 3D distance is always >= z-distance, so this is safe)
    real cutoff = SCREEN_CUTOFF_RB * d_rB;  // ~1375 reduced units

    for (int k = 0; k < ATOM_COUNT; k++) {
        real d3 = r3 - d_atomZ[k];

        // Fast z-distance check: skip if too far (avoids exp+sqrt)
        if (d3 > cutoff || d3 < -cutoff) continue;

        real d1 = r1;
        real d2 = r2;

        real dNorm2 = d1*d1 + d2*d2 + d3*d3;
        real dNorm = REAL_SQRT(dNorm2);
        real dNorm3 = dNorm2 * dNorm;

        if (dNorm3 > (real)1e-30) {
            real ddotv = d1*v1 + d2*v2 + d3*v3;
            real screening = REAL_EXP(-dNorm / d_rB);
            real emFactor = twoZAlpha * screening * sqrtFactor / dNorm3;
            dv1 -= emFactor * (d1 - v1 * ddotv);
            dv2 -= emFactor * (d2 - v2 * ddotv);
            dv3 -= emFactor * (d3 - v3 * ddotv);
        }
    }

    dydt[6] = dv1;  dydt[7] = dv2;  dydt[8] = dv3;

    // du/dt: zitter constraint
    real qr1 = q1 - r1, qr2 = q2 - r2, qr3 = q3 - r3;
    real qrNorm2 = qr1*qr1 + qr2*qr2 + qr3*qr3;
    real vdotu = v1*u1 + v2*u2 + v3*u3;

    if (qrNorm2 > (real)1e-30) {
        real zitterFactor = ((real)1.0 - vdotu) / qrNorm2;
        dydt[9]  = zitterFactor * qr1;
        dydt[10] = zitterFactor * qr2;
        dydt[11] = zitterFactor * qr3;
    } else {
        dydt[9]  = (real)0.0;
        dydt[10] = (real)0.0;
        dydt[11] = (real)0.0;
    }
}

// ============================================================================
// Device: DP853 error estimation
// ============================================================================
__device__ real dp853_estimateError(const real k[][12], const real* y0,
                                     const real* y1, real h) {
    real error1 = (real)0.0, error2 = (real)0.0;
    for (int j = 0; j < 12; j++) {
        real errSum1 = d_E1[0]*k[0][j] + d_E1[5]*k[5][j] + d_E1[6]*k[6][j]
                     + d_E1[7]*k[7][j] + d_E1[8]*k[8][j] + d_E1[9]*k[9][j]
                     + d_E1[10]*k[10][j] + d_E1[11]*k[11][j];
        real errSum2 = d_E2[0]*k[0][j] + d_E2[5]*k[5][j] + d_E2[6]*k[6][j]
                     + d_E2[7]*k[7][j] + d_E2[8]*k[8][j] + d_E2[9]*k[9][j]
                     + d_E2[10]*k[10][j] + d_E2[11]*k[11][j];

        real yScale = REAL_FMAX(REAL_FABS(y0[j]), REAL_FABS(y1[j]));
        real tol = d_absTol + d_relTol * yScale;
        real ratio1 = errSum1 / tol;
        real ratio2 = errSum2 / tol;
        error1 += ratio1 * ratio1;
        error2 += ratio2 * ratio2;
    }

    real den = error1 + (real)0.01 * error2;
    if (den <= (real)0.0) den = (real)1.0;
    return REAL_FABS(h) * error1 / REAL_SQRT((real)12.0 * den);
}

// ============================================================================
// Device: DP853 initial step size (Hairer & Wanner Algorithm 2.4)
// ============================================================================
__device__ real dp853_computeInitialStep(real t, const real* y,
                                          const real* yDot) {
    real d0 = (real)0.0, d1 = (real)0.0;
    for (int j = 0; j < 12; j++) {
        real sc = d_absTol + REAL_FABS(y[j]) * d_relTol;
        d0 += (y[j] / sc) * (y[j] / sc);
        d1 += (yDot[j] / sc) * (yDot[j] / sc);
    }
    d0 = REAL_SQRT(d0 / (real)12.0);
    d1 = REAL_SQRT(d1 / (real)12.0);

    real h0;
    if (d0 < (real)1e-5 || d1 < (real)1e-5) {
        h0 = (real)1e-6;
    } else {
        h0 = (real)0.01 * d0 / d1;
    }
    h0 = REAL_FMIN(h0, d_maxStep);

    // One Euler step to estimate second derivative
    real yTmp[12], f1[12];
    for (int j = 0; j < 12; j++) yTmp[j] = y[j] + h0 * yDot[j];
    rivas_rhs(t + h0, yTmp, f1);

    real d2 = (real)0.0;
    for (int j = 0; j < 12; j++) {
        real sc = d_absTol + REAL_FABS(y[j]) * d_relTol;
        real dd = (f1[j] - yDot[j]) / (sc * h0);
        d2 += dd * dd;
    }
    d2 = REAL_SQRT(d2 / (real)12.0);

    real h1;
    if (REAL_FMAX(d1, d2) <= (real)1e-15) {
        h1 = REAL_FMAX((real)1e-6, h0 * (real)1e-3);
    } else {
        h1 = REAL_POW((real)0.01 / REAL_FMAX(d1, d2), (real)(1.0 / 9.0));
    }

    real h = REAL_FMIN(REAL_FMIN((real)100.0 * h0, h1), d_maxStep);
    return REAL_FMAX(h, d_minStep);
}

// ============================================================================
// Device: Full DP853 adaptive integration for one electron
// ============================================================================
#define EXIT_FORWARD      0
#define EXIT_BACKWARD     1
#define EXIT_SUPERLUMINAL 2
#define EXIT_TIMEOUT      3
#define EXIT_MAXSTEPS     4

// Safety valve: max steps before giving up
#ifdef USE_FLOAT
#define MAX_STEPS 2000000   // FP32 with 1e-6 tol typically needs far fewer steps
#else
#define MAX_STEPS 5000000   // FP64 with 1e-12 tol — CPU takes ~1.2M
#endif

// (SCREEN_CUTOFF_RB defined above, before ATOM_COUNT)

struct IntegrationResult {
    int nSteps;
    int exitCode;
    real minimalDistance;
};

__device__ IntegrationResult dp853_integrate(real* y) {
    constexpr real SAFETY = (real)0.9;
    constexpr real MIN_REDUCTION = (real)0.2;
    constexpr real MAX_GROWTH = (real)10.0;
    constexpr real EXP_VAL = (real)(-1.0 / 8.0);

    // Stage vectors — ~1.5KB (double) or ~0.75KB (float) per thread
    real k[13][12];
    real yTmp[12];
    real y1[12];

    IntegrationResult result;
    result.nSteps = 0;
    result.exitCode = EXIT_TIMEOUT;
    result.minimalDistance = (real)1e30;

    real t = (real)0.0;

    // Initial RHS evaluation
    rivas_rhs(t, y, k[0]);

    // Initial step size
    real h = dp853_computeInitialStep(t, y, k[0]);

    int totalAttempts = 0;

    while (t < d_maxTime && result.nSteps < MAX_STEPS) {
        if (t + h > d_maxTime) h = d_maxTime - t;
        if (h < d_minStep) h = d_minStep;

        // Compute stages k[1]..k[12]
        for (int stage = 1; stage <= 12; stage++) {
            for (int j = 0; j < 12; j++) {
                real sum = (real)0.0;
                for (int l = 0; l < stage; l++) {
                    sum += d_A[stage-1][l] * k[l][j];
                }
                yTmp[j] = y[j] + h * sum;
            }
            rivas_rhs(t + d_C[stage-1] * h, yTmp, k[stage]);
        }

        // Propagate: y1 = y + h * sum(B[i] * k[i])
        for (int j = 0; j < 12; j++) {
            real sum = (real)0.0;
            for (int l = 0; l < 13; l++) {
                sum += d_B[l] * k[l][j];
            }
            y1[j] = y[j] + h * sum;
        }

        // Error estimation
        real error = dp853_estimateError(k, y, y1, h);

        totalAttempts++;
        if (totalAttempts > 20000000) {
            result.exitCode = EXIT_MAXSTEPS;
            return result;
        }

        if (error <= (real)1.0) {
            result.nSteps++;
            t += h;
            for (int j = 0; j < 12; j++) y[j] = y1[j];
            for (int j = 0; j < 12; j++) k[0][j] = k[12][j]; // FSAL

            // Track closest approach every 64 steps
            if ((result.nSteps & 63) == 0) {
                real rx = y[3], ry = y[4], rz = y[5];
                real xy2 = rx*rx + ry*ry;
                for (int a = 0; a < ATOM_COUNT; a++) {
                    real dz = rz - d_atomZ[a];
                    real dist = REAL_SQRT(xy2 + dz*dz);
                    if (dist < result.minimalDistance) result.minimalDistance = dist;
                }
            }

            // Forward detection
            if (y[2] > d_detectionDistance) {
                result.exitCode = EXIT_FORWARD;
                return result;
            }
            // Backward detection
            if (y[2] < -d_detectionDistance && y[8] < (real)0.0) {
                result.exitCode = EXIT_BACKWARD;
                return result;
            }
            // Superluminal check
            real v2 = y[6]*y[6] + y[7]*y[7] + y[8]*y[8];
            if (v2 > (real)0.9999) {
                result.exitCode = EXIT_SUPERLUMINAL;
                return result;
            }

            // Grow step
            real factor;
            if (error <= (real)1e-20) {
                factor = MAX_GROWTH;
            } else {
                factor = REAL_FMIN(MAX_GROWTH, REAL_FMAX(MIN_REDUCTION,
                    SAFETY * REAL_POW(error, EXP_VAL)));
            }
            h = REAL_FMIN(h * factor, d_maxStep);
        } else {
            real factor = REAL_FMAX(MIN_REDUCTION,
                SAFETY * REAL_POW(error, EXP_VAL));
            h = REAL_FMAX(h * factor, d_minStep);
        }
    }

    return result;
}

// ============================================================================
// Data transfer structs
// ============================================================================
struct ElectronInput {
    real state[12];
    double initialKE;  // keep in double for host-side accuracy
    double dxZERO;
    double psi0;
};

struct ElectronOutput {
    real finalState[12];
    double initialKE;
    int nSteps;
    int exitCode;
    real minimalDistance;
    double dxZERO;
    double psi0;
};

// ============================================================================
// CUDA kernel — one thread per electron
// ============================================================================
__global__ void integrateKernel(const ElectronInput* __restrict__ inputs,
                                ElectronOutput* __restrict__ outputs,
                                int numElectrons) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numElectrons) return;

    // Copy initial state to local memory
    real state[12];
    for (int i = 0; i < 12; i++) state[i] = inputs[idx].state[i];

    // Run full adaptive integration
    IntegrationResult result = dp853_integrate(state);

    // Write results
    for (int i = 0; i < 12; i++) outputs[idx].finalState[i] = state[i];
    outputs[idx].initialKE = inputs[idx].initialKE;
    outputs[idx].nSteps = result.nSteps;
    outputs[idx].exitCode = result.exitCode;
    outputs[idx].minimalDistance = result.minimalDistance;
    outputs[idx].dxZERO = inputs[idx].dxZERO;
    outputs[idx].psi0 = inputs[idx].psi0;
}

// ============================================================================
// Host: Generate initial conditions (replicates Electron constructor logic)
// ============================================================================
ElectronInput generateElectron(double energy, double rangeMin, double rangeMax,
                                std::mt19937& rng) {
    using namespace PhysConst;
    ElectronInput e;

    std::uniform_real_distribution<double> impactDist(rangeMin, rangeMax);
    std::uniform_real_distribution<double> signDist(0.0, 1.0);
    std::uniform_real_distribution<double> phaseDist(0.0, 2.0 * M_PI);

    double dxZERO = impactDist(rng);
    if (signDist(rng) > 0.5) dxZERO = -dxZERO;
    dxZERO /= zitterRadius;
    double dyZERO = 0.0;
    double dzZERO = startPos;

    e.initialKE = energy;
    e.dxZERO = dxZERO * zitterRadius;  // store in meters for output

    double gamma0 = energy / m0c2 + 1.0;
    double beta0 = std::sqrt(1.0 - 1.0 / (gamma0 * gamma0));
    double velocity0 = beta0;

    double Xdotx0 = 0.0, Xdoty0 = 0.0, Xdotz0 = velocity0;

    // Initialize spin axis — uniform random on sphere if "random", else fixed
    double theta0, phi0;
    if (spinRandom) {
        std::uniform_real_distribution<double> uni01(0.0, 1.0);
        theta0 = std::acos(1.0 - 2.0 * uni01(rng));
        phi0   = 2.0 * M_PI * uni01(rng);
    } else {
        theta0 = spinTheta0;
        phi0   = spinPhi0;
    }
    double psi0 = phaseDist(rng);
    e.psi0 = psi0;

    double rxZERO = std::cos(theta0)*std::cos(phi0)*std::cos(psi0) - std::sin(phi0)*std::sin(psi0);
    double ryZERO = std::cos(theta0)*std::sin(phi0)*std::cos(psi0) + std::cos(phi0)*std::sin(psi0);
    double rzZERO = -std::sin(theta0)*std::cos(psi0);

    double uxZERO = std::cos(theta0)*std::cos(phi0)*std::sin(psi0) + std::sin(phi0)*std::cos(psi0);
    double uyZERO = std::cos(theta0)*std::sin(phi0)*std::sin(psi0) - std::cos(phi0)*std::cos(psi0);
    double uzZERO = -std::sin(theta0)*std::sin(psi0);

    double vdotrZero = Xdotx0*rxZERO + Xdoty0*ryZERO + Xdotz0*rzZERO;
    double vdotuZero = Xdotx0*uxZERO + Xdoty0*uyZERO + Xdotz0*uzZERO;

    // q(0) = boosted mass position
    e.state[0] = (real)(vdotrZero*uxZERO - vdotuZero*rxZERO + dxZERO);
    e.state[1] = (real)(vdotrZero*uyZERO - vdotuZero*ryZERO + dyZERO);
    e.state[2] = (real)(vdotrZero*uzZERO - vdotuZero*rzZERO + dzZERO);

    // r(0) = boosted charge position
    e.state[3] = (real)(rxZERO - (gamma0/(1.0+gamma0))*vdotrZero*Xdotx0 + dxZERO);
    e.state[4] = (real)(ryZERO - (gamma0/(1.0+gamma0))*vdotrZero*Xdoty0 + dyZERO);
    e.state[5] = (real)(rzZERO - (gamma0/(1.0+gamma0))*vdotrZero*Xdotz0 + dzZERO);

    // v(0) = mass velocity
    e.state[6] = (real)Xdotx0;
    e.state[7] = (real)Xdoty0;
    e.state[8] = (real)Xdotz0;

    // u(0) = boosted charge velocity
    e.state[9]  = (real)((uxZERO + gamma0*Xdotx0 + (gamma0*gamma0/(1.0+gamma0))*vdotuZero*Xdotx0)
                 / (gamma0*(1.0 + vdotuZero)));
    e.state[10] = (real)((uyZERO + gamma0*Xdoty0 + (gamma0*gamma0/(1.0+gamma0))*vdotuZero*Xdoty0)
                 / (gamma0*(1.0 + vdotuZero)));
    e.state[11] = (real)((uzZERO + gamma0*Xdotz0 + (gamma0*gamma0/(1.0+gamma0))*vdotuZero*Xdotz0)
                 / (gamma0*(1.0 + vdotuZero)));

    return e;
}

// ============================================================================
// Host: compute kinetic energy from final state
// ============================================================================
double getKineticEnergy(const real* state) {
    double v2 = (double)state[6]*state[6] + (double)state[7]*state[7] + (double)state[8]*state[8];
    if (v2 >= 1.0) return -1.0;
    double gamma = 1.0 / std::sqrt(1.0 - v2);
    return (gamma - 1.0) * PhysConst::m0c2;
}

double getAngle(const real* state) {
    double a = std::atan2((double)state[2], (double)state[0]) * 180.0 / M_PI;
    if (a < 0) a += 360.0;
    return a;
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
    PhysConst::init();

    // Parse args: elektron2_cuda [numElectrons] [energyEV] [spinAxis]
    int totalSimulations = PhysConst::defaultSimulations;
    if (argc > 1) totalSimulations = std::atoi(argv[1]);
    if (totalSimulations < 1) totalSimulations = 1;
    double energyEV = PhysConst::startEnergy;
    if (argc > 2) {
        double e = std::atof(argv[2]);
        if (e > 0) energyEV = e;
    }
    if (argc > 3) PhysConst::setSpinAxis(argv[3]);

    // GPU info
    int deviceId;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&deviceId));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));

    printf("GPU: %s | SM %d.%d | %d SMs | %.0f MB VRAM\n",
           prop.name, prop.major, prop.minor,
           prop.multiProcessorCount,
           prop.totalGlobalMem / (1024.0 * 1024.0));

    printf("PARAMS | rangeMin: %.2e | rangeMax: %.2e | startEnergy: %.0f eV"
           " | spin: %s | Z: %.0f | atoms: %d | spacing: %.1f (reduced)\n",
           PhysConst::rangeMin, PhysConst::rangeMax, energyEV,
           PhysConst::spinLabel.c_str(), PhysConst::carbonProtons,
           PhysConst::atomCount, PhysConst::atomSpacing);
#ifdef USE_FLOAT
    printf("Integrator: DP853 (CUDA, FP32) | absTol: 1e-6 | relTol: 1e-6"
           " | minStep: 1e-8 | maxStep: %.1f\n", PhysConst::dp853MaxStep);
#else
    printf("Integrator: DP853 (CUDA, FP64) | absTol: %.0e | relTol: %.0e"
           " | minStep: %.0e | maxStep: %.1f\n",
           PhysConst::dp853AbsTol, PhysConst::dp853RelTol,
           PhysConst::dp853MinStep, PhysConst::dp853MaxStep);
#endif
    printf("Running %d simulations on GPU.\n\n", totalSimulations);

    // Initialize __constant__ memory
    initDeviceConstants();

    // Generate initial conditions on host
    auto totalStart = std::chrono::steady_clock::now();

    std::vector<ElectronInput> h_inputs(totalSimulations);
    std::mt19937 rng(std::random_device{}());
    for (int i = 0; i < totalSimulations; i++) {
        h_inputs[i] = generateElectron(energyEV,
            PhysConst::rangeMin, PhysConst::rangeMax, rng);
    }

    // Allocate device memory
    ElectronInput* d_inputs;
    ElectronOutput* d_outputs;
    CUDA_CHECK(cudaMalloc(&d_inputs, totalSimulations * sizeof(ElectronInput)));
    CUDA_CHECK(cudaMalloc(&d_outputs, totalSimulations * sizeof(ElectronOutput)));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_inputs, h_inputs.data(),
        totalSimulations * sizeof(ElectronInput), cudaMemcpyHostToDevice));

    // Launch kernel
    int blockSize = 64;  // conservative — each thread uses ~2KB local memory
    int gridSize = (totalSimulations + blockSize - 1) / blockSize;

    printf("Launching kernel: %d blocks x %d threads = %d threads\n",
           gridSize, blockSize, gridSize * blockSize);
    fflush(stdout);

    auto kernelStart = std::chrono::steady_clock::now();

    integrateKernel<<<gridSize, blockSize>>>(d_inputs, d_outputs, totalSimulations);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto kernelEnd = std::chrono::steady_clock::now();
    long kernelMs = std::chrono::duration_cast<std::chrono::milliseconds>(
        kernelEnd - kernelStart).count();

    // Copy results back
    std::vector<ElectronOutput> h_outputs(totalSimulations);
    CUDA_CHECK(cudaMemcpy(h_outputs.data(), d_outputs,
        totalSimulations * sizeof(ElectronOutput), cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(d_inputs));
    CUDA_CHECK(cudaFree(d_outputs));

    // Tally results
    int isNaN_total = 0, isPos_total = 0, isNeg_total = 0;
    long totalSteps = 0;

    for (int i = 0; i < totalSimulations; i++) {
        auto& o = h_outputs[i];
        if (o.exitCode == EXIT_SUPERLUMINAL) {
            isNaN_total++;
        } else {
            if (o.finalState[2] > 0) isPos_total++;
            else isNeg_total++;
        }
        totalSteps += o.nSteps;

        if (totalSimulations <= 100 ||
            (i+1) % PhysConst::progressLogEvery == 0 ||
            i == totalSimulations - 1) {
            double eOut = getKineticEnergy(o.finalState);
            const char* exitStr = (o.exitCode == EXIT_FORWARD) ? "FWD" :
                                  (o.exitCode == EXIT_BACKWARD) ? "BACK" :
                                  (o.exitCode == EXIT_SUPERLUMINAL) ? "NaN" :
                                  (o.exitCode == EXIT_MAXSTEPS) ? "MAXS" : "TIME";
            printf("[%5d] Steps: %7d | Exit: %4s | Apex: %.4e | Eout: %.4e eV\n",
                   i, o.nSteps, exitStr, (double)o.minimalDistance, eOut);
        }
    }

    auto totalEnd = std::chrono::steady_clock::now();
    long totalMs = std::chrono::duration_cast<std::chrono::milliseconds>(
        totalEnd - totalStart).count();

    printf("\n=== SUMMARY ===\n");
    printf("isNaN: %d | isPos: %d | isNeg: %d\n",
           isNaN_total, isPos_total, isNeg_total);
    printf("Total steps: %ld | Avg steps/electron: %ld\n",
           totalSteps, totalSteps / totalSimulations);
    printf("KERNEL TIME: %ld ms\n", kernelMs);
    printf("TOTAL TIME FOR %d SIMULATIONS: %ld ms\n", totalSimulations, totalMs);

    // ================================================================
    // Write full-precision results file
    // ================================================================
    {
        auto now = std::chrono::system_clock::now();
        std::time_t nowTime = std::chrono::system_clock::to_time_t(now);
        char timeBuf[64], dateBuf[64], timeFmt[64];
        std::strftime(timeBuf, sizeof(timeBuf), "%Y-%m-%d %H:%M:%S", std::localtime(&nowTime));
        std::strftime(dateBuf, sizeof(dateBuf), "%Y-%m-%d", std::localtime(&nowTime));
        std::strftime(timeFmt, sizeof(timeFmt), "%H%M%S", std::localtime(&nowTime));

        std::string resultsDir = "/mnt/c/Users/marcf/IdeaProjects/ELektron2/results/";
        char energyStr[32];
        std::snprintf(energyStr, sizeof(energyStr), "%.0f", energyEV);
        std::string resultsFile = std::string(dateBuf) + "_" + timeFmt
            + "_cuda-dp853_" + energyStr + "eV_spin" + PhysConst::spinLabel
            + "_" + std::to_string(totalSimulations) + ".dat";
        std::string resultsPath = resultsDir + resultsFile;

        std::ofstream out(resultsPath);
        if (!out.is_open()) {
            resultsPath = resultsFile;
            out.open(resultsPath);
        }
        out << std::setprecision(std::numeric_limits<double>::max_digits10);

        out << "# ELektron2 CUDA Simulation Results\n";
        out << "# Date: " << timeBuf << "\n";
        out << "# Integrator: DP853 (CUDA, GPU: " << prop.name << ")\n";
        out << "# DP853 absTol: " << PhysConst::dp853AbsTol
            << "  relTol: " << PhysConst::dp853RelTol
            << "  minStep: " << PhysConst::dp853MinStep
            << "  maxStep: " << PhysConst::dp853MaxStep << "\n";
        out << "# Kernel time: " << kernelMs << " ms\n";
        out << "# Total time: " << totalMs << " ms\n";
        out << "# Total simulations: " << totalSimulations << "\n";
        out << "# startEnergy: " << energyEV << " eV\n";
        out << "# startPos: " << PhysConst::startPos << " (reduced)\n";
        out << "# detectionDistance: " << PhysConst::detectionDistance << " (reduced)\n";
        out << "# rangeMin: " << PhysConst::rangeMin << " m\n";
        out << "# rangeMax: " << PhysConst::rangeMax << " m\n";
        out << "# spin: " << PhysConst::spinLabel << "\n";
        out << "# spinOrientation: " << PhysConst::spinLabel << "\n";
        out << "# theta0: " << PhysConst::spinTheta0 << " rad\n";
        out << "# phi0: " << PhysConst::spinPhi0 << " rad\n";
        out << "# Z: " << PhysConst::carbonProtons << "\n";
        out << "# alpha: " << PhysConst::alpha << "\n";
        out << "# reducedBohr: " << PhysConst::reducedBohr << "\n";
        out << "# zitterRadius: " << PhysConst::zitterRadius << " m\n";
        out << "# atomCount: " << PhysConst::atomCount << "\n";
        out << "# atomSpacing: " << PhysConst::atomSpacing << " (reduced) = "
            << PhysConst::atomSpacingMeters << " m\n";
        out << "# chainHalfLength: " << PhysConst::chainHalfLength << " (reduced)\n";
        out << "# maxTime: " << PhysConst::maxTime << " (reduced)\n";
        out << "# Summary: isNaN=" << isNaN_total << " isPos=" << isPos_total
            << " isNeg=" << isNeg_total << "\n";
        out << "#\n";
        out << "# Columns:\n";
        out << "# idx qx qy qz rx ry rz vx vy vz ux uy uz"
            << " energyIn_eV energyOut_eV angle_deg steps"
            << " apexCharge exitCode dxZERO_reduced psi0\n";
        out << "#\n";

        for (int i = 0; i < totalSimulations; i++) {
            auto& o = h_outputs[i];
            const real* s = o.finalState;
            double v2 = (double)s[6]*s[6] + (double)s[7]*s[7] + (double)s[8]*s[8];
            double eOut = getKineticEnergy(s);
            double angle = getAngle(s);

            out << i;
            for (int j = 0; j < 12; j++) out << " " << (double)s[j];
            out << " " << o.initialKE
                << " " << eOut
                << " " << angle
                << " " << o.nSteps
                << " " << (double)o.minimalDistance
                << " " << o.exitCode
                << " " << o.dxZERO
                << " " << o.psi0
                << "\n";
        }

        out.close();
        printf("Wrote %d electron results to %s\n", totalSimulations, resultsPath.c_str());
    }

    return 0;
}
