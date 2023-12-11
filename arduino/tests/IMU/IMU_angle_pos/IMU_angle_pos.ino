#include "Wire.h"
#include <MPU6050_light.h>

MPU6050 mpu(Wire);
unsigned long timer = 0;

void setup() {
  Serial.begin(115200);
  Wire.begin();

  byte status = mpu.begin();
  Serial.print(F("MPU6050 status: "));
  Serial.println(status);
  while(status != 0) {} // Stop if could not connect to MPU6050
  
  Serial.println(F("Calculating offsets, do not move MPU6050"));
  delay(1000);
  mpu.calcOffsets(); // Gyro and accelerometer
  Serial.println("Done!\n");
}

void loop() {
  mpu.update();
  
  if ((millis() - timer) > 100) { // Print data every 100ms
    Serial.print("Linear Acceleration X: ");
    Serial.print(mpu.getAccX());
    Serial.print("\tY: ");
    Serial.print(mpu.getAccY());
    Serial.print("\tZ: ");
    Serial.println(mpu.getAccZ());
    
    Serial.print("Position X: ");
    Serial.print(mpu.getPosX());
    Serial.print("\tY: ");
    Serial.print(mpu.getPosY());
    Serial.print("\tZ: ");
    Serial.println(mpu.getPosZ());

    timer = millis();
  }
}
