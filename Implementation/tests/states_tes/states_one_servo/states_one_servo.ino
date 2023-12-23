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
long duration1, duration2;
float distanceRB1, distanceRB2;

const int servoPin = 25;  // Change to GPIO 25 for the single servo
Servo servo;  // Change to a single servo object

// Packed data to be sent through Serial
struct SensorData {
  int angles[10];     // LIST WITH 10 INTEGERS VALUES (JOINT ANGLES)
  float distanceRB1;  // FLOAT WITH DISTANCE IN CM FROM ROBOT1
  float distanceRB2;  // FLOAT WITH DISTANCE IN CM FROM ROBOT2
  float object_pitch;
  float object_yaw;
  float object_roll;
};

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

  attachServoMotor();

  pinMode(trig1Pin, OUTPUT);
  pinMode(trig2Pin, OUTPUT);
  pinMode(echo1Pin, INPUT);
  pinMode(echo2Pin, INPUT);
}

void loop() {
  SensorData sensorData;

  getMotorAngle(sensorData.angles[0]);  // Only one servo angle
  readUltrasonicDistance(sensorData.distanceRB1, sensorData.distanceRB2);

  // Send the packed data through Serial
  Serial.write((uint8_t*)&sensorData, sizeof(sensorData));
}

// Attach the servo to its Pin in ESP32
void attachServoMotor() {
  servo.attach(servoPin);  // Attach the servo to GPIO 25
}

// Read angle for the single motor
void getMotorAngle(int &angle) {
  angle = servo.read();  // Read the angle from the single servo
}

// Read distance (cm) for each HSCR04
void readUltrasonicDistance(float &distance1, float &distance2) {
  // Clears the trigPin
  digitalWrite(trig1Pin, LOW);
  digitalWrite(trig2Pin, LOW);
  delayMicroseconds(2);
  // Sets the trigPin on HIGH state for 10 microseconds
  digitalWrite(trig1Pin, HIGH);
  digitalWrite(trig2Pin, HIGH);
  delayMicroseconds(10);

  digitalWrite(trig1Pin, LOW);
  digitalWrite(trig2Pin, LOW);
  
  // Reads the echoPin, returns the sound wave travel time in microseconds
  duration1 = pulseIn(echo1Pin, HIGH);
  duration2 = pulseIn(echo2Pin, HIGH);
  
  // Calculate the distance
  distance1 = duration1 * SOUND_SPEED/2;
  distance2 = duration2 * SOUND_SPEED/2;
}
