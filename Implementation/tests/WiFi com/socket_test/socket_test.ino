#include <WiFi.h>

const char *ssid = "PENDEJO TU";      // Your Wi-Fi SSID
const char *password = "uuuuuuuu";  // Your Wi-Fi password

const int trigPin = 32;
const int echoPin = 33;

#define SOUND_SPEED 0.034
WiFiServer server(80);

float distanceCm;
long duration;

void setup() {
  Serial.begin(115200);
  pinMode(trigPin, OUTPUT)
  pinMode(echoPin, INPUT);

  delay(10);

  // Connect to Wi-Fi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi...");
  }

  Serial.println("Connected to WiFi");

  // Print ESP32 IP address
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());

  server.begin();
}

void loop() {
  WiFiClient client = server.available();

  if (client) {
    Serial.println("New client connected");
    distanceCm = getDistance();  // Replace with your actual distance calculation function

    // Send the distance data to the client
    client.print("Distance (cm): ");
    client.print(distanceCm);
    client.print(",Distance (inch): ");
    client.println(distanceCm * 0.393701);

    client.stop();
    Serial.println("Client disconnected");
  }
}

float getDistance() {
  // Clears the trigPin
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  // Sets the trigPin on HIGH state for 10 micro seconds
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
  
  // Reads the echoPin, returns the sound wave travel time in microseconds
  duration = pulseIn(echoPin, HIGH);
  
  // Calculate the distance
  distanceCm = duration * SOUND_SPEED/2;
  
  return distanceCm

}
