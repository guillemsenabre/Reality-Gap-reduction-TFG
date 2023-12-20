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

const int servoPins[] = {26, 27, 28, 29, 30, 31, 32, 33, 34, 35};
const int numServos = 10;
Servo servos[numServos];

// Packed data to be sent through Serial
struct SensorData {
  int angles[10];
  float distanceRB1;
  float distanceRB2;
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

  attachServoMotors();

  pinMode(trig1Pin, OUTPUT);
  pinMode(trig2Pin, OUTPUT);
  pinMode(echo1Pin, INPUT);
  pinMode(echo2Pin, INPUT);
}


void loop() {

  SensorData sensorData;

  getMotorAngles(sensorData.servoAngles);
  readUltrasonicDistance(sensorData.distanceRB1, sensorData.distanceRB2);

  //what?
  Serial.write((uint8_t*)&sensorData, sizeof(sensorData));
}


// attach each servor to its Pin in ESP32
void attachServoMotors() {
  for (int i = 0; i < numServos; i++) {
    servos[i].attach(servoPins[i]);
  }
}

// Read angle for each motor
void getMotorAngles(int angles[]) {
  for (int i = 0; i < numServos; i++) {
    angles[i] = servos[i].read();
  }
}

// Read distance (cm) for each HSCR04
float readUltrasonicDistance(float &distance1, float &distance2) {
  // Clears the trigPin
  digitalWrite(trig1Pin, LOW);
  digitalWrite(trig2Pin, LOW);
  delayMicroseconds(2);
  // Sets the trigPin on HIGH state for 10 micro seconds
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
