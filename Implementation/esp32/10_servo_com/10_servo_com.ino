#include <Servo.h>

#define NUM_SERVOS 10

// Define servo pins
int servoPins[NUM_SERVOS] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

Servo servos[NUM_SERVOS];

void setup() {
  Serial.begin(115200);
  while(!Serial);

  // Attach each servo to its respective pin
  for (int i = 0; i < NUM_SERVOS; i++) {
    servos[i].attach(servoPins[i]);
  }
}

void loop() {
  // Wait for serial input with torque values for each servo
  if (Serial.available() >= NUM_SERVOS * sizeof(float)) {
    // Read torque values for each servo
    for (int i = 0; i < NUM_SERVOS; i++) {
      float torqueValue = 0.0;
      Serial.readBytes((char*)&torqueValue, sizeof(float));

      // Map torque value to servo angle (adjust mapping as needed)
      int angle = map(torqueValue, 0, 1, 0, 180);

      // Write the mapped angle to the servo
      servos[i].write(angle);
    }
  }
}
