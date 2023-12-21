#include <Servo.h>

#define SERVO1_PIN 26
#define SERVO2_PIN 27

Servo servoMotor1;
Servo servoMotor2;

void setup() {
  Serial.begin(115200);
  while(!Serial);
  servoMotor1.attach(SERVO1_PIN);
  servoMotor2.attach(SERVO2_PIN);
}

int getMotorAngle(Servo servo) {
  return servo.read();
}

void loop() {
  for (int pos = 0; pos <= 180; pos += 1) {
    servoMotor1.write(pos);
    servoMotor2.write(pos);

    int angle1 = servoMotor1.read();
    int angle2 = servoMotor2.read();

    Serial.print("Servo 1 Angle: ");
    Serial.print(angle1);
    Serial.print("\tServo 2 Angle: ");
    Serial.println(angle2);

   // int angle1 = getMotorAngle(servoMotor1);
   // int angle2 = getMotorAngle(servoMotor2);

    delay(50);

    
  }
  for (int pos = 180; pos >= 0; pos -= 1) {
    servoMotor1.write(pos);
    servoMotor2.write(pos);

    // int angle1 = getMotorAngle(servoMotor1);
    // int angle2 = getMotorAngle(servoMotor2);

    int angle1 = servoMotor1.read();
    int angle2 = servoMotor2.read();

    Serial.print("Servo 1 Angle: ");
    Serial.print(angle1);
    Serial.print("\tServo 2 Angle: ");
    Serial.println(angle2);
    
    delay(50);
  }
}
