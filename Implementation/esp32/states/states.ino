#include <Servo.h>
#include <Wire.h>
#include <MPU6050_light.h>

const int trig1Pin = 32;
const int echo1Pin = 33;
const int trig2Pin = 34;
const int echo2Pin = 35;

const int servo11Pin = 26;
const int servo12Pin = 27;
const int servo13Pin = 27;
const int servo14Pin = 27;
const int servo15Pin = 27;

const int servo21Pin = 27;
const int servo22Pin = 27;
const int servo23Pin = 27;
const int servo24Pin = 27;
const int servo25Pin = 27;

MPU6050 mpu(Wire);
unsigned long timer = 0;

#define SOUND_SPEED 0.034

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

  while(status != 0);

  mpu.calcOffsets();
  Serial.println("Mpu calibration completed!")
  
  while(!Serial);

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

void loop() {
  // put your main code here, to run repeatedly:

}
