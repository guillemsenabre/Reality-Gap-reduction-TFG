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
const int number_motors;
unsigned long previousMillis = 0;
const long interval = 30; // Adjust to change the speed

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

  for (int i = 0; i < number_motors; i++) {
    #define SER(i) i
  }

  while (!Serial) {}

  attachServoMotor();
  Serial.println("Servo attached.");

  pinMode(trig1Pin, OUTPUT);
  pinMode(trig2Pin, OUTPUT);
  pinMode(echo1Pin, INPUT);
  pinMode(echo2Pin, INPUT);
}

void loop() {
  SensorData sensorData;

  getMotorAngle(sensorData.angles[0]);  // Only one servo angle
  Serial.print("Servo Angle: ");
  Serial.println(sensorData.angles[0]);

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
  delay(100);  // Add a delay for readability in the Serial Monitor
}


// Attach the servo to its Pin in ESP32
void attachServoMotor() {
  servo.attach(servoPin);  // Attach the servo to GPIO 25
}

// Read angle for the single motor
void getMotorAngle(int &angle) {
  angle = servo.read();  // Read the angle from the single servo
  Serial.println("Reading Servo Angle...");
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