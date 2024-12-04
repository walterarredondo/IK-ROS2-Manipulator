#include <Servo.h>

// Define Servo objects for each joint
Servo joint1; // Base
Servo joint2;
Servo joint3;
Servo joint4;
Servo joint5; // Gripper or Top Joint

// Define joint pins
const int joint1Pin = 3;
const int joint2Pin = 5;
const int joint3Pin = 6;
const int joint4Pin = 9;
const int joint5Pin = 10;

void setup() {
  Serial.begin(115200);

  // Attach servos to pins
  joint1.attach(joint1Pin);
  joint2.attach(joint2Pin);
  joint3.attach(joint3Pin);
  joint4.attach(joint4Pin);
  joint5.attach(joint5Pin);

  // Initialize servos to a default position (e.g., 90 degrees)
  joint1.write(90);
  joint2.write(90);
  joint3.write(180); //limitar a 35
  joint4.write(90);
  joint5.write(90);

  Serial.println("Ready to receive commands: joint,angle");
}

void loop() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n'); // Read until newline

    int commaIndex = input.indexOf(',');
    if (commaIndex == -1) {
      Serial.println("Invalid format. Use: joint,angle");
      return;
    }

    int jointNum = input.substring(0, commaIndex).toInt();
    int angle = input.substring(commaIndex + 1).toInt();

    if (jointNum < 1 || jointNum > 5) {
      Serial.println("Invalid joint number. Must be 1-5.");
      return;
    }

    if (angle < 0 || angle > 180) {
      Serial.println("Invalid angle. Must be 0-180.");
      return;
    }

    // Move the selected joint to the specified angle
    switch (jointNum) {
      case 1:
        joint1.write(angle);
        break;
      case 2:
        joint2.write(angle);
        break;
      case 3:
        joint3.write(angle);
        break;
      case 4:
        joint4.write(angle);
        break;
      case 5:
        joint5.write(angle);
        break;
    }

    Serial.print("Moving Joint ");
    Serial.print(jointNum);
    Serial.print(" to ");
    Serial.print(angle);
    Serial.println(" degrees.");
  }
}

