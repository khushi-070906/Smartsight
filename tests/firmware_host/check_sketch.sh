#!/bin/sh
# Host syntax/type check of the whole sketch against stub headers (NOT a real ESP32 build).
cd "$(dirname "$0")"
cp ../../firmware/smartsight_cane/smartsight_cane.ino /tmp/sketch_check.cpp
sed -i '1i #include <Arduino.h>' /tmp/sketch_check.cpp
g++ -std=c++17 -fsyntax-only -Wall -Wextra -Wno-unused-parameter -Istubs -I../../firmware/smartsight_cane /tmp/sketch_check.cpp
