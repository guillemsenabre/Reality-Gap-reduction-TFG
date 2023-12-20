#include <Servo.h>
#include <Wire.h>
#include <MPU6050_light.h>

const int trig1Pin = 2;
const int echo1Pin = 4;
const int trig2Pin = 5;
const int echo2Pin = 18;

MPU6050 mpu(Wire);

unsigned long timer = 0;
const float SOUND_SPEED = 0.034;
long duration;
float distanceCm;

const int servoPins[] = {26, 27, 28, 29, 30, 31, 32, 33, 34, 35};
const int numServos = 10;
Servo servos[numServos];

void setup() {
  Serial.begin(115200);

  Wire.begin();

  byte status = mpu.begin();

  while (status != 0) {
    Serial.println("MPU initialization failed!");
    delay(1000);
    status = mpu.begin();
  }

  mpu.calcOffsets();
  Serial.println("MPU calibration completed!");

  while (!Serial) {}

  attachServoMotors();

  pinMode(trig1Pin, OUTPUT);
  pinMode(trig2Pin, OUTPUT);
  pinMode(echo1Pin, INPUT);
  pinMode(echo2Pin, INPUT);
}

void loop() {
  int angles[10];
  getMotorAngles(angles);
}




void attachServoMotors() {
  for (int i = 0; i < numServos; i++) {
    servos[i].attach(servoPins[i]);
  }
}

void getMotorAngles(int angles[]) {
  for (int i = 0; i < numServos; i++) {
    angles[i] = servos[i].read();
  }
}
