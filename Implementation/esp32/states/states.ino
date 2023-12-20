#include <Servo.h>
#include <Wire.h>
#include <MPU6050_light.h>

const int trig1Pin = 2;
const int echo1Pin = 4;
const int trig2Pin = 5;
const int echo2Pin = 18;

const int servo11Pin = 26;
const int servo12Pin = 27;
const int servo13Pin = 28;
const int servo14Pin = 29;
const int servo15Pin = 30;

const int servo21Pin = 31;
const int servo22Pin = 32;
const int servo23Pin = 33;
const int servo24Pin = 34;
const int servo25Pin = 35;

MPU6050 mpu(Wire);

unsigned long timer = 0;
const float SOUND_SPEED = 0.034;
long duration;
float distanceCm;

Servo servoMotor11;
Servo servoMotor12;
Servo servoMotor13;
Servo servoMotor14;
Servo servoMotor15;

Servo servoMotor21;
Servo servoMotor22;
Servo servoMotor23;
Servo servoMotor24;
Servo servoMotor25;

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
  servoMotor11.attach(servo11Pin);
  servoMotor12.attach(servo12Pin);
  servoMotor13.attach(servo13Pin);
  servoMotor14.attach(servo14Pin);
  servoMotor15.attach(servo15Pin);

  servoMotor21.attach(servo21Pin);
  servoMotor22.attach(servo22Pin);
  servoMotor23.attach(servo23Pin);
  servoMotor24.attach(servo24Pin);
  servoMotor25.attach(servo25Pin);
}

void getMotorAngles(int angles[]) {
  angles[0] = servoMotor11.read();
  angles[1] = servoMotor12.read();
  angles[2] = servoMotor13.read();
  angles[3] = servoMotor14.read();
  angles[4] = servoMotor15.read();
  
  angles[5] = servoMotor21.read();
  angles[6] = servoMotor22.read();
  angles[7] = servoMotor23.read();
  angles[8] = servoMotor24.read();
  angles[9] = servoMotor25.read();
}
