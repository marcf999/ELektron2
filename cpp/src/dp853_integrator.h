#pragma once

// Self-contained Dormand-Prince 8(5,3) adaptive ODE integrator
// Exact coefficient match to Apache commons-math3 DormandPrince853Integrator
// Reference: Hairer, Norsett, Wanner "Solving ODEs I", Table 5.2
// No external dependencies. CUDA-portable.

#include <cmath>
#include <algorithm>
#include <cstring>

namespace dp853 {

inline const double S6 = std::sqrt(6.0);

// Time fractions c_2..c_13 (c_1 = 0 implicit, stages 2-13)
inline const double C[12] = {
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

// Butcher tableau A[i][j]: row i for stage i+2, column j for stage j+1
// Lower triangular, zero-initialized
inline const double A[12][12] = {
    // stage 2
    {(12.0 - 2.0*S6) / 135.0},
    // stage 3
    {(6.0 - S6) / 180.0, (6.0 - S6) / 60.0},
    // stage 4
    {(6.0 - S6) / 120.0, 0.0, (6.0 - S6) / 40.0},
    // stage 5
    {(462.0 + 107.0*S6) / 3000.0, 0.0,
     (-402.0 - 197.0*S6) / 1000.0, (168.0 + 73.0*S6) / 375.0},
    // stage 6
    {1.0/27.0, 0.0, 0.0, (16.0 + S6) / 108.0, (16.0 - S6) / 108.0},
    // stage 7
    {19.0/512.0, 0.0, 0.0,
     (118.0 + 23.0*S6) / 1024.0, (118.0 - 23.0*S6) / 1024.0,
     -9.0/512.0},
    // stage 8
    {13772.0/371293.0, 0.0, 0.0,
     (51544.0 + 4784.0*S6) / 371293.0, (51544.0 - 4784.0*S6) / 371293.0,
     -5688.0/371293.0, 3072.0/371293.0},
    // stage 9
    {58656157643.0/93983540625.0, 0.0, 0.0,
     (-1324889724104.0 - 318801444819.0*S6) / 626556937500.0,
     (-1324889724104.0 + 318801444819.0*S6) / 626556937500.0,
     96044563816.0/3480871875.0, 5682451879168.0/281950621875.0,
     -165125654.0/3796875.0},
    // stage 10
    {8909899.0/18653125.0, 0.0, 0.0,
     (-4521408.0 - 1137963.0*S6) / 2937500.0,
     (-4521408.0 + 1137963.0*S6) / 2937500.0,
     96663078.0/4553125.0, 2107245056.0/137915625.0,
     -4913652016.0/147609375.0, -78894270.0/3880452869.0},
    // stage 11
    {-20401265806.0/21769653311.0, 0.0, 0.0,
     (354216.0 + 94326.0*S6) / 112847.0,
     (354216.0 - 94326.0*S6) / 112847.0,
     -43306765128.0/5313852383.0, -20866708358144.0/1126708119789.0,
     14886003438020.0/654632330667.0,
     35290686222309375.0/14152473387134411.0,
     -1477884375.0/485066827.0},
    // stage 12
    {39815761.0/17514443.0, 0.0, 0.0,
     (-3457480.0 - 960905.0*S6) / 551636.0,
     (-3457480.0 + 960905.0*S6) / 551636.0,
     -844554132.0/47026969.0, 8444996352.0/302158619.0,
     -2509602342.0/877790785.0,
     -28388795297996250.0/3199510091356783.0,
     226716250.0/18341897.0, 1371316744.0/2131383595.0},
    // stage 13 (= B weights, FSAL)
    {104257.0/1920240.0, 0.0, 0.0, 0.0, 0.0,
     3399327.0/763840.0, 66578432.0/35198415.0,
     -1674902723.0/288716400.0, 54980371265625.0/176692375811392.0,
     -734375.0/4826304.0, 171414593.0/851261400.0, 137909.0/3084480.0}
};

// Propagation weights (8th order)
inline const double B[13] = {
    104257.0/1920240.0, 0.0, 0.0, 0.0, 0.0,
    3399327.0/763840.0, 66578432.0/35198415.0,
    -1674902723.0/288716400.0, 54980371265625.0/176692375811392.0,
    -734375.0/4826304.0, 171414593.0/851261400.0, 137909.0/3084480.0,
    0.0
};

// Error weights E1 (primary: 8th vs 5th order)
// Only indices 0,5,6,7,8,9,10,11 are nonzero
inline const double E1[12] = {
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

// Error weights E2 (secondary)
inline const double E2[12] = {
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

} // namespace dp853


template<int N>
class DP853Integrator {
public:
    double absTol, relTol, minStep, maxStep;
    int nSteps = 0;
    int nRejected = 0;

    DP853Integrator(double minStep, double maxStep, double absTol, double relTol)
        : absTol(absTol), relTol(relTol), minStep(minStep), maxStep(maxStep) {}

    /// Integrate from t0 toward tEnd.
    /// rhs: void(double t, const double* y, double* dydt)
    /// callback: bool(double t, const double* y) — return false to stop early
    /// Returns the time reached.
    template<typename RHS, typename Callback>
    double integrate(RHS&& rhs, double t0, double* y, double tEnd, Callback&& callback) {
        using namespace dp853;

        double t = t0;
        double k[13][N];
        double yTmp[N];
        double y1[N];

        // Initial evaluation
        rhs(t, y, k[0]);

        // Initial step size (Hairer & Wanner Algorithm 2.4)
        double h = computeInitialStep(rhs, t, y, k[0], tEnd - t0);

        while (t < tEnd) {
            if (t + h > tEnd) h = tEnd - t;
            if (h < minStep) h = minStep;

            // Compute stages k[1]..k[12]
            for (int stage = 1; stage <= 12; stage++) {
                for (int j = 0; j < N; j++) {
                    double sum = 0;
                    for (int l = 0; l < stage; l++) {
                        sum += A[stage-1][l] * k[l][j];
                    }
                    yTmp[j] = y[j] + h * sum;
                }
                rhs(t + C[stage-1] * h, yTmp, k[stage]);
            }

            // y1 = y + h * sum(B[i] * k[i])
            for (int j = 0; j < N; j++) {
                double sum = 0;
                for (int l = 0; l < 13; l++) {
                    sum += B[l] * k[l][j];
                }
                y1[j] = y[j] + h * sum;
            }

            // Error estimation (matches commons-math3 exactly)
            double error = estimateError(k, y, y1, h);

            if (error <= 1.0) {
                // Accept step
                nSteps++;
                t += h;
                std::memcpy(y, y1, N * sizeof(double));
                std::memcpy(k[0], k[12], N * sizeof(double));  // FSAL

                if (!callback(t, y)) return t;

                // Grow step
                double factor;
                if (error <= 1e-20) {
                    factor = MAX_GROWTH;
                } else {
                    factor = std::min(MAX_GROWTH, std::max(MIN_REDUCTION,
                        SAFETY * std::pow(error, EXP)));
                }
                h = std::min(h * factor, maxStep);
            } else {
                // Reject step
                nRejected++;
                double factor = std::max(MIN_REDUCTION,
                    SAFETY * std::pow(error, EXP));
                h = std::max(h * factor, minStep);
            }
        }

        return t;
    }

private:
    static constexpr double SAFETY = 0.9;
    static constexpr double MIN_REDUCTION = 0.2;
    static constexpr double MAX_GROWTH = 10.0;
    static constexpr double EXP = -1.0 / 8.0;

    // Error estimation matching commons-math3 DormandPrince853Integrator.estimateError()
    double estimateError(const double k[][N], const double* y0,
                         const double* y1, double h) const {
        using namespace dp853;

        double error1 = 0, error2 = 0;
        for (int j = 0; j < N; j++) {
            // Weighted error sums (only nonzero indices: 0,5,6,7,8,9,10,11)
            double errSum1 = E1[0]*k[0][j] + E1[5]*k[5][j] + E1[6]*k[6][j]
                           + E1[7]*k[7][j] + E1[8]*k[8][j] + E1[9]*k[9][j]
                           + E1[10]*k[10][j] + E1[11]*k[11][j];
            double errSum2 = E2[0]*k[0][j] + E2[5]*k[5][j] + E2[6]*k[6][j]
                           + E2[7]*k[7][j] + E2[8]*k[8][j] + E2[9]*k[9][j]
                           + E2[10]*k[10][j] + E2[11]*k[11][j];

            double yScale = std::max(std::abs(y0[j]), std::abs(y1[j]));
            double tol = absTol + relTol * yScale;
            double ratio1 = errSum1 / tol;
            double ratio2 = errSum2 / tol;
            error1 += ratio1 * ratio1;
            error2 += ratio2 * ratio2;
        }

        double den = error1 + 0.01 * error2;
        if (den <= 0.0) den = 1.0;

        return std::abs(h) * error1 / std::sqrt(N * den);
    }

    // Initial step size estimation (Hairer & Wanner, Algorithm 2.4)
    template<typename RHS>
    double computeInitialStep(RHS&& rhs, double t, const double* y,
                              const double* yDot, double span) const {
        double d0 = 0, d1 = 0;
        for (int j = 0; j < N; j++) {
            double sc = absTol + std::abs(y[j]) * relTol;
            d0 += (y[j] / sc) * (y[j] / sc);
            d1 += (yDot[j] / sc) * (yDot[j] / sc);
        }
        d0 = std::sqrt(d0 / N);
        d1 = std::sqrt(d1 / N);

        double h0;
        if (d0 < 1e-5 || d1 < 1e-5) {
            h0 = 1e-6;
        } else {
            h0 = 0.01 * d0 / d1;
        }
        h0 = std::min(h0, maxStep);

        // One Euler step to estimate second derivative
        double yTmp[N], f1[N];
        for (int j = 0; j < N; j++) yTmp[j] = y[j] + h0 * yDot[j];
        rhs(t + h0, yTmp, f1);

        double d2 = 0;
        for (int j = 0; j < N; j++) {
            double sc = absTol + std::abs(y[j]) * relTol;
            double dd = (f1[j] - yDot[j]) / (sc * h0);
            d2 += dd * dd;
        }
        d2 = std::sqrt(d2 / N);

        double h1;
        if (std::max(d1, d2) <= 1e-15) {
            h1 = std::max(1e-6, h0 * 1e-3);
        } else {
            h1 = std::pow(0.01 / std::max(d1, d2), 1.0 / 9.0);  // 1/(order+1)
        }

        double h = std::min({100.0 * h0, h1, maxStep});
        return std::max(h, minStep);
    }
};
