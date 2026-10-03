#pragma once
// Minimal Arduino-ESP32 3.x API surface used by smartsight_cane.ino, for host -fsyntax-only checks.
// Signatures follow the real core; this checks OUR code's syntax/types, not the real library behaviour.
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <math.h>
#include <algorithm>
#define ARDUINO 10607
#define IRAM_ATTR
#define HIGH 1
#define LOW 0
#define INPUT 0
#define OUTPUT 1
#define INPUT_PULLUP 2
#define CHANGE 3
#define degrees(r) ((r) * 57.29577951308232)
#define constrain(x, lo, hi) ((x) < (lo) ? (lo) : ((x) > (hi) ? (hi) : (x)))
using std::max;
using std::min;
class String {
 public:
  std::string s;
  String() {}
  String(const char *c) : s(c) {}
  String(const std::string &c) : s(c) {}
  int indexOf(const char *x) const { auto p = s.find(x); return p == std::string::npos ? -1 : (int)p; }
  size_t length() const { return s.size(); }
  String &operator+=(char c) { s += c; return *this; }
  String &operator=(const char *c) { s = c; return *this; }
  const char *c_str() const { return s.c_str(); }
};
struct HardwareSerial {
  void begin(unsigned long) {}
  void println(const char *) {}
  int available() { return 0; }
  int read() { return 0; }
};
extern HardwareSerial Serial;
void pinMode(uint8_t, uint8_t);
int digitalRead(uint8_t);
void digitalWrite(uint8_t, uint8_t);
uint32_t millis();
uint32_t micros();
void delay(uint32_t);
void delayMicroseconds(uint32_t);
void analogReadResolution(uint8_t);
uint32_t analogReadMilliVolts(uint8_t);
bool ledcAttach(uint8_t pin, uint32_t freq, uint8_t res);
bool ledcWrite(uint8_t pin, uint32_t duty);
uint32_t ledcWriteTone(uint8_t pin, uint32_t freq);
int digitalPinToInterrupt(int);
void attachInterrupt(int, void (*)(void), int);
