//prints when wrong ingredient, can track orders of on off 
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
#define LED_wrong 47   // LED for "Wrong ingredient!"
#define LED_success 48 // LED for successful sequence

// Threshold for capacitive touch
int threshold = 500;

// Variables to track sensor states
bool sensorStates[4] = {false, false, false, false}; // Tracks if sensors are touched
bool ledStates[4] = {false, false, false, false};    // Tracks LED on/off states

void setup() {
  // Initialize serial communication for debugging
  Serial.begin(9600);

  // Set LED pins as output
  pinMode(LED_onion, OUTPUT);
  pinMode(LED_tomato, OUTPUT);
  pinMode(LED_spinach, OUTPUT);
  pinMode(LED_egg, OUTPUT);
  pinMode(LED_wrong, OUTPUT);
  pinMode(LED_success, OUTPUT);

  // Disable auto-calibration for sensors to improve touch consistency
  cs_onion.set_CS_AutocaL_Millis(0xFFFFFFFF);
  cs_tomato.set_CS_AutocaL_Millis(0xFFFFFFFF);
  cs_spinach.set_CS_AutocaL_Millis(0xFFFFFFFF);
  cs_egg.set_CS_AutocaL_Millis(0xFFFFFFFF);
}

void loop() {
  // Read serial data from the Python code
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();  // Remove any trailing whitespace or newlines

    // Handle the incoming command
    if (command == "WRONG") {
      digitalWrite(LED_wrong, HIGH);  // Turn on the "Wrong ingredient!" LED
      delay(2000);                   // Keep it on for 2 seconds
      digitalWrite(LED_wrong, LOW);  // Turn off the LED
    } else if (command == "SUCCESS") {
      digitalWrite(LED_success, HIGH);  // Turn on the success LED
      delay(2000);                      // Keep it on for 2 seconds
      digitalWrite(LED_success, LOW);   // Turn off the success LED
    }
  }

  // Capacitive touch handling
  long sensorValues[4] = {
    cs_onion.capacitiveSensor(30),
    cs_tomato.capacitiveSensor(30),
    cs_spinach.capacitiveSensor(30),
    cs_egg.capacitiveSensor(30)
  };

  // Check capacitive sensors and toggle LEDs on touch
  for (int i = 0; i < 4; i++) {
    if (sensorValues[i] > threshold) {
      if (!sensorStates[i]) {
        // Sensor just touched, toggle state
        sensorStates[i] = true;
        ledStates[i] = !ledStates[i];  // Toggle LED state

        if (ledStates[i]) {
          digitalWrite(getLEDPin(i), HIGH);  // Turn on the LED
          Serial.print("TOUCH ");
          Serial.println(i + 1); // Notify the Python script of the touch
        } else {
          digitalWrite(getLEDPin(i), LOW);   // Turn off the LED
          Serial.print("RELEASE ");
          Serial.println(i + 1); // Notify the Python script of the release
        }
      }
    } else {
      // Reset sensor state when not touched
      sensorStates[i] = false;
    }
  }

  delay(50); // Stability delay
}

// Function to get the LED pin based on object index
int getLEDPin(int index) {
  switch (index) {
    case 0: return LED_onion;
    case 1: return LED_tomato;
    case 2: return LED_spinach;
    case 3: return LED_egg;
    default: return -1; // Invalid index
  }
}

