#include <Servo.h>

#define SERVO_PIN 15

Servo servoMotor;

void setup() {
  servoMotor.attach(SERVO_PIN);
}

void loop() {
  for (int pos = 0; pose <= 180, pos += 1) {
    servoMotor.write(pos);
    delay(15);
  }

  for (int pos = 180; pos >= 0; pos -= 1) {
    servoMotor.write(pos);
    delay(15);
  }
}
