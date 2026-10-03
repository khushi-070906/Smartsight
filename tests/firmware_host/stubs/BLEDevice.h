#pragma once
#include "Arduino.h"
#include <stdint.h>
#define NIMBLE_STUB 1
class BLEDescriptor {};
class BLE2902 : public BLEDescriptor {};
class BLECharacteristic {
 public:
  static const uint32_t PROPERTY_NOTIFY = 1, PROPERTY_WRITE = 2, PROPERTY_WRITE_NR = 4;
  void setValue(uint8_t *, size_t);
  void notify();
  void addDescriptor(BLEDescriptor *);
  void setCallbacks(class BLECharacteristicCallbacks *);
  String getValue();
};
class BLECharacteristicCallbacks {
 public:
  virtual ~BLECharacteristicCallbacks() {}
  virtual void onWrite(BLECharacteristic *) {}
};
class BLEAdvertising {
 public:
  void addServiceUUID(const char *);
  void setScanResponse(bool);
  void start();
};
class BLEService {
 public:
  BLECharacteristic *createCharacteristic(const char *, uint32_t);
  void start();
};
class BLEServer;
class BLEServerCallbacks {
 public:
  virtual ~BLEServerCallbacks() {}
  virtual void onConnect(BLEServer *) {}
  virtual void onDisconnect(BLEServer *) {}
};
class BLEServer {
 public:
  void setCallbacks(BLEServerCallbacks *);
  BLEService *createService(const char *);
  BLEAdvertising *getAdvertising();
};
class BLEDevice {
 public:
  static void init(const char *);
  static BLEServer *createServer();
  static BLEAdvertising *getAdvertising();
  static void startAdvertising();
};
