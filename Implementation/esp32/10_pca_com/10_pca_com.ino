#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pca9685 = Adafruit_PWMServoDriver();

const int number_motors = 10;
const int rot_limit_1 = 20;
const int rot_limit_2 = 160;
const int SERVOMIN = 80;  // Adjust these values based on your servo
const int SERVOMAX = 600;

void setup() {
  Serial.begin(115200);
  while (!Serial);

  pca9685.begin();
  pca9685.setPWMFreq(50);
}

void loop() {
  // Wait for serial input with torque values for each servo
  if (Serial.available() >= number_motors * sizeof(float)) {
    // Read raw bytes into an array
    byte torqueBytes[number_motors * sizeof(float)];
    Serial.readBytes(torqueBytes, number_motors * sizeof(float));

    // Interpret bytes as float values using pointer casting.
    float torqueValues[number_motors];
    for (int i = 0; i < number_motors; i++) {
      torqueValues[i] = *((float*)&torqueBytes[i * sizeof(float)]);
    }

    // Map torque values to servo angles and control servos
    for (int i = 0; i < number_motors; i++) {
      int posDegrees = map(torqueValues[i], 0, 1, rot_limit_1, rot_limit_2);
      int pwm = map(posDegrees, rot_limit_1, rot_limit_2, SERVOMIN, SERVOMAX);
      pca9685.setPWM(i, 0, pwm);
    }
  }
}
