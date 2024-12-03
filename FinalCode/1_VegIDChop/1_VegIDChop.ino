#include <CapacitiveSensor.h>
#include <Servo.h>

// Define capacitive sensors (pins may vary based on your setup)
CapacitiveSensor cs_onion = CapacitiveSensor(13, 45);   // Sensor for Onion (pin 45)
CapacitiveSensor cs_tomato = CapacitiveSensor(13, 40);  // Sensor for Tomato (pin 40)
CapacitiveSensor cs_spinach = CapacitiveSensor(13, 42); // Sensor for Spinach (pin 42)
CapacitiveSensor cs_egg = CapacitiveSensor(13, 44);     // Sensor for Egg (pin 44)

// Define LED pins
#define LED_onion 53   // Onion
#define LED_tomato 52  // Tomato
#define LED_spinach 50 // Spinach
#define LED_egg 51     // Egg

// Servo setup
Servo servo;
int servoPin = 38;  // Pin connected to servo
int servoPosition = 0;  // Initial servo position

// Threshold for capacitive touch
int threshold = 500;

// Debounce timing
unsigned long debounceDelay = 300; // 300ms debounce delay
unsigned long lastTouchTime[4] = {0, 0, 0, 0}; // Store the last time a sensor was touched

// States and counters
bool sensorStates[4] = {false, false, false, false}; // Tracks if sensors are touched
int touchCount[4] = {0, 0, 0, 0}; // Tracks the number of touches for each sensor
bool sequenceOneComplete = false; // Tracks whether Snippet 1 logic is complete

void setup() {
  // Initialize serial communication
  Serial.begin(9600);

  // Set LED pins as output
  pinMode(LED_onion, OUTPUT);
  pinMode(LED_tomato, OUTPUT);
  pinMode(LED_spinach, OUTPUT);
  pinMode(LED_egg, OUTPUT);

  // Attach servo to its pin
  servo.attach(servoPin);
  servo.write(servoPosition); // Start at the initial position

  // Disable auto-calibration for sensors
  cs_onion.set_CS_AutocaL_Millis(0xFFFFFFFF);
  cs_tomato.set_CS_AutocaL_Millis(0xFFFFFFFF);
  cs_spinach.set_CS_AutocaL_Millis(0xFFFFFFFF);
  cs_egg.set_CS_AutocaL_Millis(0xFFFFFFFF);
}

void loop() {
  if (!sequenceOneComplete) {
    handleSequenceOne(); // Handle the first sequence (LEDs and capacitive touch)
  } else {
    handleSequenceTwo(); // Handle the second sequence (Servo movement)
  }
}

void handleSequenceOne() {
  // Read incoming commands from Python
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();

    if (command.startsWith("ON")) {
      String object = command.substring(3);
      int ledPin = getLEDPin(object);

      if (ledPin != -1) {
        blinkLED(ledPin);
        Serial.println("LED ON");
      }
    } else if (command == "SEQUENCE_ONE_COMPLETE") {
      sequenceOneComplete = true;
    }
  }

  // Read capacitive sensor values
  long sensorValues[4] = {
    cs_onion.capacitiveSensor(30),
    cs_tomato.capacitiveSensor(30),
    cs_spinach.capacitiveSensor(30),
    cs_egg.capacitiveSensor(30)
  };

  // Handle capacitive touch
  for (int i = 0; i < 4; i++) {
    if (sensorValues[i] > threshold) {
      if (!sensorStates[i] && (millis() - lastTouchTime[i] > debounceDelay)) {
        sensorStates[i] = true;
        lastTouchTime[i] = millis();
        touchCount[i]++;
        Serial.print("TOUCH ");
        Serial.println(i + 1);
        digitalWrite(getLEDPinByIndex(i), HIGH);
      }
    } else {
      sensorStates[i] = false;
    }
  }
}

void handleSequenceTwo() {
  // Move servo when touch is detected
  long sensorValues[4] = {
    cs_onion.capacitiveSensor(30),
    cs_tomato.capacitiveSensor(30),
    cs_spinach.capacitiveSensor(30),
    cs_egg.capacitiveSensor(30)
  };

  for (int i = 0; i < 4; i++) {
    if (sensorValues[i] > threshold) {
      if (!sensorStates[i] && (millis() - lastTouchTime[i] > debounceDelay)) {
        sensorStates[i] = true;
        lastTouchTime[i] = millis();
        Serial.print("SERVO MOVE ");
        Serial.println(i + 1);
        moveServo();
      }
    } else {
      sensorStates[i] = false;
    }
  }
}

void blinkLED(int ledPin) {
  for (int i = 0; i < 3; i++) {
    digitalWrite(ledPin, HIGH);
    delay(300);
    digitalWrite(ledPin, LOW);
    delay(300);
  }
}

void moveServo() {
  for (int pos = 45; pos <= 90; pos += 5) {
    servo.write(pos);
    delay(50);
  }
  for (int pos = 90; pos >= 45; pos -= 5) {
    servo.write(pos);
    delay(50);
  }
    for (int pos = 45; pos <= 90; pos += 5) {
    servo.write(pos);
    delay(50);
  }
  for (int pos = 90; pos >= 45; pos -= 5) {
    servo.write(pos);
    delay(50);
  }
    for (int pos = 45; pos <= 90; pos += 5) {
    servo.write(pos);
    delay(50);
  }
  for (int pos = 90; pos >= 45; pos -= 5) {
    servo.write(pos);
    delay(50);
  }
}

int getLEDPin(String object) {
  if (object == "Onion") return LED_onion;
  if (object == "Tomato") return LED_tomato;
  if (object == "Spinach") return LED_spinach;
  if (object == "Egg") return LED_egg;
  return -1;
}

int getLEDPinByIndex(int index) {
  switch (index) {
    case 0: return LED_onion;
    case 1: return LED_tomato;
    case 2: return LED_spinach;
    case 3: return LED_egg;
  }
  return -1;
}
