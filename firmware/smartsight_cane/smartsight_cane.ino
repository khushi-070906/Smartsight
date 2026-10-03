/*
 * SmartSight cane unit (10) firmware - ESP32 (110)
 *
 * Board: ESP32 Dev Module, Arduino-ESP32 core 3.x (LEDC API below is 3.x).
 * Libraries: "Adafruit_VL53L0X" (Library Manager). BLE + Wire are built into the core.
 *
 * Wiring (ESP32 DevKit V1):
 *   HC-SR04 (102)   TRIG -> GPIO5      ECHO -> GPIO18 THROUGH a 1k/2k divider (5V echo!)  VCC 5V
 *   VL53L0X (104)   SDA  -> GPIO21     SCL  -> GPIO22   (I2C 0x29)                         VCC 3.3V
 *   MPU6050 (106)   SDA  -> GPIO21     SCL  -> GPIO22   (I2C 0x68, AD0 low)               VCC 3.3V
 *   Vibration (114) GPIO25 -> 1k -> NPN/logic MOSFET gate, motor to VBAT, flyback diode
 *   Buzzer (116)    GPIO26 -> passive piezo (via transistor if loud piezo)
 *   SOS btn (112)   GPIO27 -> button -> GND (internal pull-up)
 *   Talk btn (opt)  GPIO14 -> button -> GND (push-to-talk for voice queries)
 *   Battery (118)   VBAT -> 100k -> GPIO34 -> 100k -> GND
 *
 * Responsibilities (everything heavy runs on the phone/laptop, never here):
 *   - first-line obstacle alert from ultrasonic + ToF, works with NO phone connected
 *   - fall detection at 100 Hz (same state machine as smartsight/fall_detect.py)
 *   - SOS button (press = SOS, press again during countdown = cancel)
 *   - 10 Hz sensor stream + events over BLE Nordic UART (also mirrored on USB serial)
 *   - executes haptic / SOS-state commands from the app
 *   - if SOS is pressed while the phone is disconnected: local siren (cannot dispatch)
 */

#include <string.h>
#include <Wire.h>
#include <Adafruit_VL53L0X.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

// ----------------------------------------------------------------- pins
#define PIN_TRIG 5
#define PIN_ECHO 18
#define PIN_VIB 25
#define PIN_BUZ 26
#define PIN_SOS 27
#define PIN_TALK 14
#define PIN_BAT 34

// ----------------------------------------------------------------- tuning
#define NEAR_URGENT_CM 50      // continuous vibration + fast beeps
#define NEAR_SOON_CM 100       // pulsed vibration
#define APP_HAPTIC_TTL_MS 3000 // app pattern expires unless refreshed (link-drop safety)
#define STREAM_MS 100          // 10 Hz sensor packets
#define DEBOUNCE_MS 40

static const char *DEVICE_NAME = "SmartSight-Cane";
#define NUS_SERVICE "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
#define NUS_RX "6e400002-b5a3-f393-e0a9-e50e24dcca9e"
#define NUS_TX "6e400003-b5a3-f393-e0a9-e50e24dcca9e"

// ----------------------------------------------------------------- state
Adafruit_VL53L0X tof;
bool tofOk = false;
BLECharacteristic *txChar = nullptr;
volatile bool bleConnected = false;

volatile uint32_t echoStart = 0, echoUs = 0;
volatile bool echoFresh = false;
int usCm = -1, tofMm = -1;

float ax = 0, ay = 0, az = 1, yawDeg = 0, gzBias = 0;
float batV = 0;

enum Level { L_OFF = 0, L_OK = 1, L_SOON = 2, L_URGENT = 3, L_SOS = 4 };
Level localLevel = L_OFF, appLevel = L_OFF;
uint32_t appLevelAt = 0;

enum SosUi { SOS_IDLE, SOS_PENDING, SOS_SENT, SOS_FAILED, SOS_LOCAL_SIREN };
SosUi sosUi = SOS_IDLE;
uint32_t sosUiAt = 0;

String bleLine, serialLine;   // separate buffers: BLE callbacks run on another task

// ----------------------------------------------------------------- output
void emit(const char *json) {
  Serial.println(json);
  if (bleConnected && txChar) {
    size_t n = strlen(json);
    // BLE notifications are MTU-limited; send in 20-byte chunks, app reassembles on '\n'
    for (size_t i = 0; i < n; i += 20) {
      size_t len = min((size_t)20, n - i);
      txChar->setValue((uint8_t *)json + i, len);
      txChar->notify();
    }
    txChar->setValue((uint8_t *)"\n", 1);
    txChar->notify();
  }
}

void emitEvent(const char *ev, int cm = -1) {
  char buf[64];
  if (cm >= 0) snprintf(buf, sizeof(buf), "{\"t\":\"e\",\"ev\":\"%s\",\"cm\":%d}", ev, cm);
  else snprintf(buf, sizeof(buf), "{\"t\":\"e\",\"ev\":\"%s\"}", ev);
  emit(buf);
}

// ----------------------------------------------------------------- app commands
Level parseLevel(const String &s) {
  if (s.indexOf("\"urgent\"") >= 0) return L_URGENT;
  if (s.indexOf("\"soon\"") >= 0) return L_SOON;
  if (s.indexOf("\"sos\"") >= 0) return L_SOS;
  if (s.indexOf("\"ok\"") >= 0) return L_OK;
  return L_OFF;
}

void handleCommand(const String &line) {
  if (line.indexOf("\"c\":\"hap\"") >= 0) {
    appLevel = parseLevel(line);
    appLevelAt = millis();
  } else if (line.indexOf("\"c\":\"sos\"") >= 0) {
    if (line.indexOf("\"pending\"") >= 0) sosUi = SOS_PENDING;
    else if (line.indexOf("\"sent\"") >= 0) sosUi = SOS_SENT;
    else if (line.indexOf("\"failed\"") >= 0) sosUi = SOS_FAILED;
    else sosUi = SOS_IDLE;
    sosUiAt = millis();
  }
}

void feedRx(String &buf, const uint8_t *data, size_t len) {
  for (size_t i = 0; i < len; i++) {
    char c = (char)data[i];
    if (c == '\n') { handleCommand(buf); buf = ""; }
    else if (buf.length() < 200) buf += c;
  }
}

class ServerCB : public BLEServerCallbacks {
  void onConnect(BLEServer *) override { bleConnected = true; }
  void onDisconnect(BLEServer *s) override {
    bleConnected = false;
    appLevel = L_OFF;
    s->getAdvertising()->start();
  }
};

class RxCB : public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic *c) override {
    String v = c->getValue();
    feedRx(bleLine, (const uint8_t *)v.c_str(), v.length());
  }
};

void setupBle() {
  BLEDevice::init(DEVICE_NAME);
  BLEServer *server = BLEDevice::createServer();
  server->setCallbacks(new ServerCB());
  BLEService *svc = server->createService(NUS_SERVICE);
  txChar = svc->createCharacteristic(NUS_TX, BLECharacteristic::PROPERTY_NOTIFY);
  txChar->addDescriptor(new BLE2902());
  BLECharacteristic *rx = svc->createCharacteristic(
      NUS_RX, BLECharacteristic::PROPERTY_WRITE | BLECharacteristic::PROPERTY_WRITE_NR);
  rx->setCallbacks(new RxCB());
  svc->start();
  BLEAdvertising *adv = BLEDevice::getAdvertising();
  adv->addServiceUUID(NUS_SERVICE);
  adv->setScanResponse(true);
  BLEDevice::startAdvertising();
}

// ----------------------------------------------------------------- ultrasonic (non-blocking)
void IRAM_ATTR onEcho() {
  if (digitalRead(PIN_ECHO)) echoStart = micros();
  else { echoUs = micros() - echoStart; echoFresh = true; }
}

void pollUltrasonic() {
  static uint32_t lastTrig = 0;
  static bool waiting = false;
  if (echoFresh) {
    echoFresh = false;
    waiting = false;
    uint32_t us = echoUs;
    usCm = (us > 100 && us < 25000) ? (int)(us / 58) : -1;   // 25 ms ~ 4.3 m max
  }
  if (millis() - lastTrig >= 60) {                            // HC-SR04 needs >= 60 ms cycle
    if (waiting) usCm = -1;                                   // previous ping never returned
    waiting = true;
    lastTrig = millis();
    digitalWrite(PIN_TRIG, HIGH);
    delayMicroseconds(10);
    digitalWrite(PIN_TRIG, LOW);
  }
}

void pollTof() {
  if (!tofOk || !tof.isRangeComplete()) return;
  uint16_t mm = tof.readRangeResult();
  tofMm = (mm > 30 && mm < 2000) ? mm : -1;                   // VL53L0X useful range ~2 m
}

// ----------------------------------------------------------------- MPU6050 (raw registers)
#define MPU 0x68
void mpuWrite(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU); Wire.write(reg); Wire.write(val); Wire.endTransmission();
}

bool mpuRead(float &gx, float &gy, float &gz) {
  Wire.beginTransmission(MPU);
  Wire.write(0x3B);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(MPU, 14) != 14) return false;
  int16_t r[7];
  for (int i = 0; i < 7; i++) r[i] = (Wire.read() << 8) | Wire.read();
  ax = r[0] / 4096.0f; ay = r[1] / 4096.0f; az = r[2] / 4096.0f;   // +-8 g
  gx = r[4] / 65.5f; gy = r[5] / 65.5f; gz = r[6] / 65.5f;         // +-500 dps
  return true;
}

void setupMpu() {
  mpuWrite(0x6B, 0x00);   // wake
  mpuWrite(0x1A, 0x03);   // DLPF 44 Hz
  mpuWrite(0x1B, 0x08);   // gyro +-500 dps
  mpuWrite(0x1C, 0x10);   // accel +-8 g (impacts exceed +-2 g)
  delay(100);
  float gx, gy, gz, sum = 0;
  int n = 0;
  for (int i = 0; i < 300; i++) { if (mpuRead(gx, gy, gz)) { sum += gz; n++; } delay(3); }
  gzBias = n ? sum / n : 0;                                 // keep the cane still at boot
}

#include "fall_detector.h"
FallDetector fall;

// ----------------------------------------------------------------- buttons
struct Button {
  uint8_t pin; bool stable = true, last = true; uint32_t changedAt = 0;
  bool pressed() {                       // true once per press (falling edge, debounced)
    bool r = digitalRead(pin);
    if (r != last) { last = r; changedAt = millis(); }
    if (millis() - changedAt > DEBOUNCE_MS && r != stable) { stable = r; return !stable; }
    return false;
  }
};
Button sosBtn{PIN_SOS}, talkBtn{PIN_TALK};

// ----------------------------------------------------------------- actuators
void buzz(int freq) { ledcWriteTone(PIN_BUZ, freq); }
void vib(int duty) { ledcWrite(PIN_VIB, duty); }

void driveActuators() {
  uint32_t now = millis();
  if (appLevel != L_OFF && now - appLevelAt > APP_HAPTIC_TTL_MS) appLevel = L_OFF;
  Level lvl = max(localLevel, appLevel);
  bool fastBlink = (now / 150) % 2, slowBlink = (now / 400) % 2;

  // SOS UI has priority on the buzzer
  int tone = 0;
  switch (sosUi) {
    case SOS_PENDING: tone = ((now - sosUiAt) % 1000 < 120) ? 2000 : 0; break;              // 1 beep/s countdown
    case SOS_SENT:                                                                          // 3 chirps
      tone = ((now - sosUiAt) < 900 && ((now - sosUiAt) / 150) % 2 == 0) ? 2600 : 0;
      if (now - sosUiAt > 1000) sosUi = SOS_IDLE;
      break;
    case SOS_FAILED:
    case SOS_LOCAL_SIREN:                                                                   // siren
      tone = 1400 + ((now / 4) % 1200);
      if (now - sosUiAt > 30000) sosUi = SOS_IDLE;
      break;
    default:
      if (lvl == L_URGENT) tone = fastBlink ? 2400 : 0;
      break;
  }
  buzz(tone);

  int duty = 0;
  if (lvl == L_URGENT || lvl == L_SOS) duty = 255;
  else if (lvl == L_SOON) duty = slowBlink ? 200 : 0;
  else if (lvl == L_OK) duty = ((now % 2000) < 80) ? 150 : 0;
  vib(duty);
}

// ----------------------------------------------------------------- setup / loop
void setup() {
  Serial.begin(115200);
  pinMode(PIN_TRIG, OUTPUT);
  pinMode(PIN_ECHO, INPUT);
  attachInterrupt(digitalPinToInterrupt(PIN_ECHO), onEcho, CHANGE);
  pinMode(PIN_SOS, INPUT_PULLUP);
  pinMode(PIN_TALK, INPUT_PULLUP);
  analogReadResolution(12);
  ledcAttach(PIN_VIB, 20000, 8);      // 20 kHz PWM: inaudible motor drive
  ledcAttach(PIN_BUZ, 2000, 8);

  Wire.begin(21, 22, 400000);
  setupMpu();
  tofOk = tof.begin();
  if (tofOk) tof.startRangeContinuous(50);
  else Serial.println("{\"t\":\"log\",\"msg\":\"VL53L0X not found\"}");

  setupBle();
  vib(200); delay(150); vib(0);       // boot cue
}

void loop() {
  static uint32_t lastImu = 0, lastStream = 0, lastNear = 0;
  uint32_t now = millis();

  // 100 Hz IMU + fall detection
  if (now - lastImu >= 10) {
    float dt = (now - lastImu) / 1000.0f;
    lastImu = now;
    float gx, gy, gz;
    if (mpuRead(gx, gy, gz)) {
      yawDeg += (gz - gzBias) * dt;
      if (yawDeg > 360) yawDeg -= 360; else if (yawDeg < 0) yawDeg += 360;
      if (fall.update((int32_t)now, ax, ay, az)) emitEvent("fall");
    }
  }

  pollUltrasonic();
  pollTof();

  // first-line local obstacle alert - independent of the phone/AI pipeline
  int nearest = 10000;
  if (usCm > 0) nearest = usCm;
  if (tofMm > 0) nearest = min(nearest, tofMm / 10);
  Level prev = localLevel;
  localLevel = nearest < NEAR_URGENT_CM ? L_URGENT : nearest < NEAR_SOON_CM ? L_SOON : L_OFF;
  if (localLevel > prev && now - lastNear > 1500) { emitEvent("near", nearest); lastNear = now; }

  // buttons
  if (sosBtn.pressed()) {
    if (sosUi == SOS_PENDING) { emitEvent("sos_cancel"); sosUi = SOS_IDLE; }
    else if (bleConnected) emitEvent("sos_button");
    else { sosUi = SOS_LOCAL_SIREN; sosUiAt = now; emitEvent("sos_button"); }   // no phone: siren
  }
  if (talkBtn.pressed()) emitEvent("talk");

  // USB serial commands (bench mode, same protocol as BLE RX)
  while (Serial.available()) { uint8_t c = Serial.read(); feedRx(serialLine, &c, 1); }

  // 10 Hz sensor stream
  if (now - lastStream >= STREAM_MS) {
    lastStream = now;
    batV = analogReadMilliVolts(PIN_BAT) * 2 / 1000.0f;
    char buf[160];
    snprintf(buf, sizeof(buf),
             "{\"t\":\"s\",\"us\":%d,\"tof\":%d,\"ax\":%.2f,\"ay\":%.2f,\"az\":%.2f,\"yaw\":%.1f,\"bat\":%.2f}",
             usCm, tofMm, ax, ay, az, yawDeg, batV);
    emit(buf);
  }

  driveActuators();
}
