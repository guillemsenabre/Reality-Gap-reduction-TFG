#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pca9685 = Adafruit_PWMServoDriver(0x40);

// Ticks: Range from 0 to 4095 --> 2048 ticks would be 50% Duty Cycle
// This determines the PWM pulse width
#define SERVOMIN  80
#define SERVOMAX  600

unsigned long previousMillis = 0;
const long interval = 30;  // Adjust to change the speed

const int number_motors = 10;

const int rot_limit_1 = 20;
const int rot_limit_2 = 160;

void setup() {
  Serial.begin(115200);
  Serial.println("PCA9685 Servo Test");
  pca9685.begin();
  pca9685.setPWMFreq(50);

  // Define servo constants
  for (int i = 0; i < number_motors; i++) {
    #define SER(i) i
  }
}

void loop() {
  unsigned long currentMillis = millis();

  // Move Motor 0 and Motor 1 simultaneously
  for (int posDegrees = rot_limit_1; posDegrees <= rot_limit_2; posDegrees++) {
    for (int i = 0; int i < number_motors; i++) {
      pwm0 = map(posDegrees, 0, 180, SERVOMIN, SERVOMAX);
      pca9685.setPWM(SER0, 0, pwm0);
      Serial.print(pca9685.getServoValue(SER(i)));
      Serial.print("\t");
    }
    Serial.println();

    // Check if the specified interval has elapsed
    if (currentMillis - previousMillis >= interval) {
      previousMillis = currentMillis;
      delay(500); // Allow some time for the PCA9685 to process the command
    }
  }
}


// RESET ANGLES TO 90 DEGREES / RESET ANGLES TO RANDOM POSITION 
// INCLUDE THIS CODE TO STATES.IO
// abs(LAST_ACTION_VALUE - NEW_ACTION_VALUE) --> THE BIGGER, THE MORE TIME TO MOVE. IF BIG abs(10-170) = 160, with the same
//amount of time to move as abs(50-60) = 10, it will go very fast. Either this or reduce velocity of everything, so the maximum
//range is not dangerous.