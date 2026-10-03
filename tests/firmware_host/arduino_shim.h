#pragma once
// Host stand-ins for the few Arduino macros fall_detector.h uses.
#include <math.h>
#include <algorithm>
#define degrees(r) ((r) * 57.29577951308232)
#define constrain(x, lo, hi) ((x) < (lo) ? (lo) : ((x) > (hi) ? (hi) : (x)))
using std::max;
using std::min;
