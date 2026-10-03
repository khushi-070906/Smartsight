#pragma once
#include <stdint.h>
#include <stddef.h>
struct TwoWire {
  bool begin(int sda = -1, int scl = -1, uint32_t f = 0);
  void beginTransmission(uint8_t);
  uint8_t endTransmission(bool stop = true);
  size_t write(uint8_t);
  uint8_t requestFrom(uint8_t, size_t);
  int read();
};
extern TwoWire Wire;
