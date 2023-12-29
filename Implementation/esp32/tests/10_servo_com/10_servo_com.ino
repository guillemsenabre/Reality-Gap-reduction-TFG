#include <Servo.h>

#define NUM_SERVOS 10

// Define servo pins
int servoPins[NUM_SERVOS] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

Servo servos[NUM_SERVOS];

void setup() {
  Serial.begin(115200);
  while (!Serial);

  // Attach each servo to its respective pin
  for (int i = 0; i < NUM_SERVOS; i++) {
    servos[i].attach(servoPins[i]);
  }
}

void loop() {
  // Wait for serial input with torque values for each servo
  if (Serial.available() >= NUM_SERVOS * sizeof(float)) {
    // Read raw bytes into an array
    byte torqueBytes[NUM_SERVOS * sizeof(float)];
    Serial.readBytes(torqueBytes, NUM_SERVOS * sizeof(float));

    // Interpret bytes as float values using pointer casting.
    float torqueValues[NUM_SERVOS];
    for (int i = 0; i < NUM_SERVOS; i++) {
      torqueValues[i] = *((float*)&torqueBytes[i * sizeof(float)]);
    }

    // Map torque values to servo angles and control servos
    for (int i = 0; i < NUM_SERVOS; i++) {
      int angle = map(torqueValues[i], 0, 1, 0, 120);
      servos[i].write(angle);
    }
  }
}
