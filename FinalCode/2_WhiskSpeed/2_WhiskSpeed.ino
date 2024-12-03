// motion detection - arduino code
// Define LED pins
const int LED_gesture1 = 48;
const int LED_gesture2 = 47;
const int LED_gesture3 = 49;

void setup() {
  // Initialize LED pins as OUTPUT
  pinMode(LED_gesture1, OUTPUT);
  pinMode(LED_gesture2, OUTPUT);
  pinMode(LED_gesture3, OUTPUT);

  // Start serial communication
  Serial.begin(9600);
}

void loop() {
  // Check if data is available to read
  if (Serial.available() > 0) {
    // Read the incoming byte (a command from Python)
    char command = Serial.read();

    // Control LEDs based on the received command
    if (command == '1') {
      digitalWrite(LED_gesture1, HIGH);  // Turn on LED_gesture1
      digitalWrite(LED_gesture2, LOW);   // Turn off LED_gesture2
      digitalWrite(LED_gesture3, LOW);   // Turn off LED_gesture3
    } 
    else if (command == '2') {
      digitalWrite(LED_gesture1, LOW);   // Turn off LED_gesture1
      digitalWrite(LED_gesture2, HIGH);  // Turn on LED_gesture2
      digitalWrite(LED_gesture3, LOW);   // Turn off LED_gesture3
    } 
    else if (command == '3') {
      digitalWrite(LED_gesture1, LOW);   // Turn off LED_gesture1
      digitalWrite(LED_gesture2, LOW);   // Turn off LED_gesture2
      digitalWrite(LED_gesture3, HIGH);  // Turn on LED_gesture3
    } 
    else if (command == '0') {
      // Turn off all LEDs if needed
      digitalWrite(LED_gesture1, LOW);
      digitalWrite(LED_gesture2, LOW);
      digitalWrite(LED_gesture3, LOW);
    }
  }
}
