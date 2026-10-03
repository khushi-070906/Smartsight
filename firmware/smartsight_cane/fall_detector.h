#pragma once
// Fall detector - port of smartsight/fall_detect.py. Keep the two in sync;
// tests/test_firmware_parity.py compiles this header on the host and runs
// the same vectors as the Python tests.
#include <math.h>
#include <string.h>
#ifndef ARDUINO
#include "../../tests/firmware_host/arduino_shim.h"
#endif

// Port of smartsight/fall_detect.py - keep the two in sync.
struct FallDetector {
  // times are int32 milliseconds (millis()), NOT float seconds: a 32-bit float
  // loses millisecond resolution after ~1 day of uptime.
  float impactG = 2.2, freefallG = 0.45, stillStdG = 0.08;
  float tiltDeg = 50, tiltDegFF = 35, alpha = 0.02;
  int32_t settleMs = 800, stillWinMs = 1500, timeoutMs = 4000, ffLookbackMs = 600;
  int state = 0;                  // 0 idle, 1 settling, 2 checking
  float gref[3] = {0, 0, 1}, gimp[3] = {0, 0, 1};
  int32_t tImpact = 0, lastLowG = 0;
  bool haveLowG = false, hadFF = false;
  double sumM = 0, sumM2 = 0, sv[3] = {0, 0, 0};
  int n = 0;

  static float angleDeg(const float a[3], const float b[3]) {
    float na = sqrtf(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]), nb = sqrtf(b[0]*b[0] + b[1]*b[1] + b[2]*b[2]);
    if (na < 1e-6f || nb < 1e-6f) return 0;
    float c = (a[0]*b[0] + a[1]*b[1] + a[2]*b[2]) / (na * nb);
    return degrees(acosf(constrain(c, -1.0f, 1.0f)));
  }

  // nowMs: millis() cast to int32 (wraps after 24 days; differences stay correct).
  bool update(int32_t nowMs, float x, float y, float z) {
    float m = sqrtf(x*x + y*y + z*z);
    if (m < freefallG) { lastLowG = nowMs; haveLowG = true; }
    if (state == 0) {
      if (fabsf(m - 1.0f) < 0.1f) { gref[0] += alpha*(x-gref[0]); gref[1] += alpha*(y-gref[1]); gref[2] += alpha*(z-gref[2]); }
      if (m > impactG) {
        state = 1; tImpact = nowMs; memcpy(gimp, gref, sizeof(gimp));
        hadFF = haveLowG && (nowMs - lastLowG) <= ffLookbackMs;
        sumM = sumM2 = 0; sv[0] = sv[1] = sv[2] = 0; n = 0;
      }
      return false;
    }
    if (nowMs - tImpact > timeoutMs) { state = 0; return false; }
    if (state == 1) { if (nowMs - tImpact >= settleMs) state = 2; return false; }
    sumM += m; sumM2 += (double)m*m; sv[0] += x; sv[1] += y; sv[2] += z; n++;
    if (nowMs - tImpact < settleMs + stillWinMs) return false;
    state = 0;
    double mean = sumM / n, sd = sqrt(max(0.0, sumM2 / n - mean*mean));
    float mv[3] = {(float)(sv[0]/n), (float)(sv[1]/n), (float)(sv[2]/n)};
    return sd < stillStdG && angleDeg(gimp, mv) > (hadFF ? tiltDegFF : tiltDeg);
  }
};
