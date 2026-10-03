// Reads "t ax ay az" lines on stdin, prints "FALL <t>" per detection.
#include <cstdio>
#include "../../firmware/smartsight_cane/fall_detector.h"
int main() {
  FallDetector d;
  double t, x, y, z;
  while (scanf("%lf %lf %lf %lf", &t, &x, &y, &z) == 4)
    if (d.update((int32_t)(t * 1000.0 + 0.5), (float)x, (float)y, (float)z)) printf("FALL %.2f\n", t);
  return 0;
}
