#has motion control webcam
import os
import subprocess
import sys
import time
import serial
import tkinter as tk
import threading
import torch
import mediapipe as mp
import cv2
from PIL import Image, ImageTk

# Arduino Serial Port Configuration
ARDUINO_PORT = 'COM5'  # Adjust as necessary
BAUD_RATE = 9600

# Serial Communication with Arduino
arduino = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)

# Define LEDs and capacitive sensors mapping
OBJECT_LED_MAP = {
    "Onion": 1,
    "Tomato": 2,
    "Spinach": 3,
    "Egg": 4
}

# Gesture recognition model and parameters
input_size = 63  # Example: 21 landmarks (x, y, z for each)
hidden_size = 128
num_classes = 3  # 3 gesture classes

# Gesture Model Definition (from second snippet)
class GestureModel(torch.nn.Module):
    def __init__(self):
        super(GestureModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load your pre-trained gesture model
gesture_model = GestureModel()
gesture_model.load_state_dict(torch.load("gesture_model.pth"))
gesture_model.eval()

# Mediapipe Hands Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# GUI setup
root = tk.Tk()
root.title("Object Recognition and Gesture Control")

# Create a Scrollable Text Widget to emulate a terminal
text_box = tk.Text(root, width=80, height=20, wrap=tk.WORD)
text_box.pack(padx=20, pady=10)

# Add Canvas for Webcam
webcam_canvas = tk.Canvas(root, width=640, height=480)  # Adjust size as needed
webcam_canvas.pack(padx=20, pady=10)
webcam_canvas.place_forget()  # Hide webcam canvas initially

# Webcam control variables
webcam_active = False
cap = None  # OpenCV VideoCapture object

def print_to_gui(message):
    """Print the message to the GUI text box."""
    text_box.insert(tk.END, message + '\n')
    text_box.yview(tk.END)

def send_command_to_arduino(command):
    """Send a command to the Arduino."""
    arduino.write(f"{command}\n".encode())
    #print_to_gui(f"Sent to Arduino: {command.strip()}")

def run_object_recognition():
    """Run the object recognition subprocess."""
    script_path = os.path.join(os.path.dirname(__file__), 'testObjects.py')

    if not os.path.exists(script_path):
        print_to_gui(f"Error: {script_path} not found.")
        return None

    print_to_gui("Running object recognition...")
    try:
        process = subprocess.Popen([sys.executable, script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate(timeout=20)
        identified_object = stdout.strip()

        if identified_object in OBJECT_LED_MAP:
            return identified_object
        else:
            print_to_gui(f"Unrecognized object: {identified_object}")
            return None
    except subprocess.TimeoutExpired:
        process.kill()
        print_to_gui("Error: Object recognition timed out.")
        return None

def wait_for_touch(object_name):
    """Wait for capacitive touch for the specified object."""
    print_to_gui(f"Waiting for touch input for {object_name}...")

    while True:
        try:
            arduino_input = arduino.readline().decode().strip()
            if arduino_input.startswith(f"TOUCH {OBJECT_LED_MAP[object_name]}"):
                #print_to_gui(f"Touch detected for {object_name}.")
                return True
            time.sleep(0.1)
        except Exception as e:
            print_to_gui(f"Error: {e}")
            return False

def start_gesture_recognition():
    """Start the webcam feed and gesture recognition."""
    global cap, webcam_active
    webcam_active = True
    cap = cv2.VideoCapture(0)  # Start webcam feed
    if not cap.isOpened():
        print_to_gui("Error: Unable to access webcam.")
        return
    webcam_canvas.place(x=20, y=300)  # Show the canvas
    update_webcam_feed()  # Start the webcam feed and gesture recognition

def stop_webcam():
    """Stop the webcam feed."""
    global cap, webcam_active
    webcam_active = False
    if cap:
        cap.release()
        cap = None
    webcam_canvas.delete("all")  # Clear the canvas

def update_webcam_feed():
    """Update the webcam feed in the GUI with gesture recognition."""
    global webcam_active, cap
    if not webcam_active or not cap:
        return
    
    ret, frame = cap.read()
    if ret:
        # Process frame with MediaPipe Hands
        result = hands.process(frame)
        gesture_message = None  # Initialize message to display

        if result.multi_hand_landmarks:
            landmarks = []
            for lm in result.multi_hand_landmarks[0].landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            if len(landmarks) == input_size:
                # Convert landmarks to tensor and predict gesture
                landmarks_tensor = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0)
                predictions = gesture_model(landmarks_tensor)
                gesture_id = torch.argmax(predictions, dim=1).item()

                # Map gesture IDs to specific messages
                if gesture_id == 0:
                    gesture_message = "Good form!"
                elif gesture_id == 1:
                    gesture_message = "Tuck in your fingers!"
                elif gesture_id == 3:
                    gesture_message = "Tuck in your fingers!"
                else:
                    gesture_message = f"Gesture {gesture_id}"

                # Display the gesture prediction on the webcam feed
                cv2.putText(frame, gesture_message, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                #print_to_gui(gesture_message)  # Print the message to the GUI

        # Convert frame to RGB format for display in Tkinter
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert frame to ImageTk format
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        # Update canvas with new frame
        webcam_canvas.imgtk = imgtk  # Keep a reference to avoid garbage collection
        webcam_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)

    # Schedule the next frame update
    if webcam_active:
        root.after(10, update_webcam_feed)


def main():
    """Main logic for object recognition and gesture control."""
    for _ in range(4):  # Repeat the process for 4 objects
        identified_object = run_object_recognition()
        if not identified_object:
            print_to_gui("Skipping due to object recognition failure.")
            continue

        print_to_gui(f"Object Identified: {identified_object}")
        send_command_to_arduino(f"ON {identified_object}")

        #if wait_for_touch(identified_object):
        #    print_to_gui("Time to chop ingredients!")

    send_command_to_arduino("SEQUENCE_ONE_COMPLETE")
    print_to_gui("Ingredients ready. Time to chop! Starting webcam with gesture recognition...")
    start_gesture_recognition()  # Start gesture recognition after Sequence One

def run_main_thread():
    """Run the main process in a separate thread to avoid blocking the GUI."""
    try:
        main()
    except Exception as e:
        print_to_gui(f"Error during execution: {e}")

# Start the main logic in a separate thread
thread = threading.Thread(target=run_main_thread)
thread.daemon = True
thread.start()

# Start the GUI main loop
root.mainloop()
