#pragma once
#include <stdint.h>
class Adafruit_VL53L0X {
 public:
  bool begin();
  bool startRangeContinuous(uint16_t period_ms = 50);
  bool isRangeComplete();
  uint16_t readRangeResult();
};
