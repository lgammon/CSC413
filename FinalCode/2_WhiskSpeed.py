"motion detection classifcation - python"
import mediapipe as mp
import cv2
import numpy as np
import time
import serial
from collections import deque

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Initialize serial communication with Arduino
arduino = serial.Serial('COM5', 9600)  # Replace 'COM5' with the appropriate port
time.sleep(2)  # Wait for Arduino to initialize

# Initialize video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera not detected.")
    exit()

# Initialize variables
previous_position = None
current_time = None
speed_values = deque(maxlen=10)  # For smoothing speed

# Define a smoothing function
def smooth_speed(new_speed):
    speed_values.append(new_speed)
    return sum(speed_values) / len(speed_values)

# Function to calculate distance between two points
def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Function to send speed data to Arduino
def send_speed_to_arduino(speed):
    if speed < 100:
        print("Sending command '1' to Arduino")
        arduino.write(b'2')  # Command for LED_gesture1
    elif 100 <= speed <= 300:
        print("Sending command '2' to Arduino")
        arduino.write(b'1')  # Command for LED_gesture2
    else:
        print("Sending command '3' to Arduino")
        arduino.write(b'3')  # Command for LED_gesture3

# Function to display speed status on the screen
def display_speed_status(frame, speed):
    if speed < 100:
        status = "Too slow!"
    elif 100 <= speed <= 300:
        status = "Good!"
    else:
        status = "Too fast!"

    # Display the status on the frame
    cv2.putText(frame, status, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Flip frame for natural movement
    frame = cv2.flip(frame, 1)

    # Convert frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Process hand landmarks if detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract hand center (average of landmarks)
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            hand_center = (int(np.mean(x_coords) * frame.shape[1]), int(np.mean(y_coords) * frame.shape[0]))

            # Calculate speed
            if previous_position is not None:
                time_diff = time.time() - current_time
                distance = calculate_distance(previous_position, hand_center)
                speed = distance / time_diff if time_diff > 0 else 0
                smoothed_speed = smooth_speed(speed)

                # Send speed to Arduino
                send_speed_to_arduino(smoothed_speed)

                # Display speed on the screen
                cv2.putText(frame, f"Speed: {smoothed_speed:.2f} px/sec", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Display speed status on the screen
                display_speed_status(frame, smoothed_speed)

            # Update previous position and time
            previous_position = hand_center
            current_time = time.time()

    # Show the frame
    cv2.imshow("Gesture Speed Detection", frame)

    # Key controls
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Esc key to exit
        break

cap.release()
cv2.destroyAllWindows()
