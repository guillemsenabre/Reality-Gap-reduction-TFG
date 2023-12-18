#include <Servo.h>

#define SERVO1_PIN 26
#define SERVO2_PIN 27

Servo servoMotor1;
Servo servoMotor2;

void setup() {
  servoMotor1.attach(SERVO1_PIN);
  servoMotor2.attach(SERVO2_PIN);
}

void loop() {
  for (int pos = 0; pos <= 180; pos += 1) {
    servoMotor.write(pos);

    delay(15);
  }

  for (int pos = 180; pos >= 0; pos -= 1) {
    servoMotor.write(pos);
    delay(15);
  }
}
