#include <Arduino.h>

#define LED 2

void setup() {
  Serial.begin(115200);
  pinMode(LED, OUTPUT);
}

void loop() {
  for (int i = 0; i < 3; i++) {
    digitalWrite(LED, HIGH);
    delay(1000);
    digitalWrite(LED, LOW);
    delay(1000);
  }

  // Store the message in a variable
  String message = "Hello, World!";

  // Print the message to the Serial Monitor
  Serial.println(message);
  delay(1000); // Delay for 1 second between repetitions
}
