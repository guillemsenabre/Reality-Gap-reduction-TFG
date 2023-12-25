#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pca9685 = Adafruit_PWMServoDriver(0x40);

// Ticks: Range from 0 to 4095 --> 2048 ticks would be 50% Duty Cycle
// This determines the PWM pulse width
#define SERVOMIN  80
#define SERVOMAX  600

#define SER0  12
#define SER1  0

int pwm0;
int pwm1;

unsigned long previousMillis = 0;
const long interval = 30;  // Adjust to change the speed

void setup() {
  Serial.begin(115200);
  Serial.println("PCA9685 Servo Test");
  pca9685.begin();
  pca9685.setPWMFreq(50);
}

void loop() {
  unsigned long currentMillis = millis();

  // Move Motor 0 and Motor 1 simultaneously
  for (int posDegrees = 0; posDegrees <= 180; posDegrees++) {
    pwm0 = map(posDegrees, 0, 180, SERVOMIN, SERVOMAX);
    //pwm1 = map(posDegrees, 180, 0, SERVOMIN, SERVOMAX);

    pca9685.setPWM(SER0, 0, pwm0);
    //pca9685.setPWM(SER1, 0, pwm1);

    Serial.print("Motor 0 = ");
    Serial.print(posDegrees);
    Serial.print("\tMotor 1 = ");
    Serial.println(180 - posDegrees);

    delay(100);

    // Check if the specified interval has elapsed
    if (currentMillis - previousMillis >= interval) {
      previousMillis = currentMillis;
      delay(500); // Allow some time for the PCA9685 to process the command
    }
  }
}
