#include <Wire.h>
#include <MPU6050_light.h>
#include <Adafruit_PWMServoDriver.h>

const int trig1Pin = 2;
const int echo1Pin = 4;
const int trig2Pin = 5;
const int echo2Pin = 18;

MPU6050 mpu(Wire);
Adafruit_PWMServoDriver pca9685 = Adafruit_PWMServoDriver(0x40);

// HCSR04 Values
unsigned long timer = 0;
const float SOUND_SPEED = 0.034;
long duration1, duration2;
float distanceRB1, distanceRB2;

// PCA9685 AND SERVO VALUES
const int number_motors = 10;
const int rot_limit_1 = 20;
const int rot_limit_2 = 160;
unsigned long previousMillis = 0;
const long interval = 30; // Adjust to change the speed
const int SERVOMIN = 80;
const int SERVOMAX = 600;

// Packed data to be sent through Serial
struct SensorData {
  int angles[number_motors];     // LIST WITH n INTEGERS VALUES (JOINT ANGLES)
  float distanceRB1;  // FLOAT WITH DISTANCE IN CM FROM ROBOT1
  float distanceRB2;  // FLOAT WITH DISTANCE IN CM FROM ROBOT2
  float object_pitch;
  float object_yaw;
  float object_roll;
};

void setup() {
  Serial.begin(115200);
  Serial.println("Initializing...");

  // MPU INIT:
  Wire.begin();
  byte status = mpu.begin();

  while (status != 0) {
    Serial.println("MPU initialization failed!");
    delay(1000);
    status = mpu.begin();
  }

  mpu.calcOffsets();
  Serial.println("MPU calibration completed!");

  // PCA AND SERVOS INIT
  pca9685.begin();
  pca9685.setPWMFreq(50);
  defineServoMotor()

  delay(500);

  while (!Serial) {}

  pinMode(trig1Pin, OUTPUT);
  pinMode(trig2Pin, OUTPUT);
  pinMode(echo1Pin, INPUT);
  pinMode(echo2Pin, INPUT);
}

void loop() {
  SensorData sensorData;

  if (Serial.available() >= number_motors * sizeof(float)) {
    // Read raw bytes into an array
    byte torqueBytes[number_motors * sizeof(float)];
    Serial.readBytes(torqueBytes, number_motors * sizeof(float));

    // Interpret bytes as float values using pointer casting.
    float torqueValues[number_motors];
    for (int i = 0; i < number_motors; i++) {
      torqueValues[i] = *((float*)&torqueBytes[i * sizeof(float)]);
    }

    // Moving motors based on ddpg torque values.
    moveMotors(sensorData.angles, torqueValues);
  }

  Serial.print("Servo Angles: ");
  for (int i = 0; i < number_motors; i++) {
    Serial.print(sensorData.angles[i]);
    Serial.print("\t");
  }
  Serial.println();

  readUltrasonicDistance(sensorData.distanceRB1, sensorData.distanceRB2);
  Serial.print("Distance RB1: ");
  Serial.println(sensorData.distanceRB1);
  Serial.print("Distance RB2: ");
  Serial.println(sensorData.distanceRB2);

  readOrientation(sensorData.object_pitch, sensorData.object_yaw, sensorData.object_roll);
  Serial.print("Object Pitch: ");
  Serial.println(sensorData.object_pitch);
  Serial.print("Object Yaw: ");
  Serial.println(sensorData.object_yaw);
  Serial.print("Object Roll: ");
  Serial.println(sensorData.object_roll);

  // Send the packed data through Serial
  Serial.write((uint8_t*)&sensorData, sizeof(sensorData));
  

}


// define n servo objects
void defineServoMotor() {
  for (int i = 0; i < number_motors; i++) {
    #define SER(i) i
  }
}

// Map torque values to servo angles and control servos
void moveMotors(int angles[], float torqueValues[]) {
  for (int i = 0; i < number_motors; i++) {
    int posDegrees = map(torqueValues[i], 0, 1, rot_limit_1, rot_limit_2);
    int pwm = map(posDegrees, rot_limit_1, rot_limit_2, SERVOMIN, SERVOMAX);
    pca9685.setPWM(i, 0, pwm);
  }
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

void readOrientation(float &pitch, float &yaw, float &roll) {
  mpu.update();
  pitch = mpu.getAngleX();
  yaw = mpu.getAngleY();
  roll = mpu.getAngleZ();
}