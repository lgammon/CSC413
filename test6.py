import torch
import torch.nn as nn
import numpy as np
import mediapipe as mp
import cv2

# Define the GestureModel class
class GestureModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(GestureModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Define model input size, hidden size, and number of output classes
input_size = 63  # Example for 21 landmarks (x, y, z for each)
hidden_size = 128
num_classes = 2  # Adjust based on your number of gesture classes

# Initialize the model
model = GestureModel(input_size, hidden_size, num_classes)

# Load your trained PyTorch model
model.load_state_dict(torch.load("gesture_model.pth"))
model.eval()

# Function to predict the gesture from hand landmarks
def predict_gesture(hand_landmarks):
    # Convert the hand landmarks to a tensor and process it
    input_data = torch.tensor(hand_landmarks).float().unsqueeze(0)  # Shape should be (1, input_size)
    
    # Get the model's prediction
    with torch.no_grad():  # No gradient tracking needed during inference
        outputs = model(input_data)  # Forward pass
    
    # Get the predicted class (highest score)
    _, predicted_class = torch.max(outputs, 1)
    
    return predicted_class.item()

# Initialize MediaPipe Hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Function to process video frames and get hand landmarks
def get_hand_landmarks(image):
    # Convert image to RGB (MediaPipe uses RGB format)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Perform hand detection and landmark extraction
    results = hands.process(image_rgb)
    
    # Check if any hands are detected
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Convert landmarks to a flat list of 63 values (21 landmarks * 3 coordinates each)
            hand_landmarks = []
            for lm in landmarks.landmark:
                hand_landmarks.extend([lm.x, lm.y, lm.z])  # Flatten the 3D coordinates
            return hand_landmarks
    
    return None  # If no hand detected, return None

# Gesture names for display (adjust based on your actual gestures)
gesture_names = ['Gesture 1', 'Gesture 2', 'Gesture 3', 'Gesture 4']

# Start webcam capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Get hand landmarks from the frame
    hand_landmarks = get_hand_landmarks(frame)
    
    if hand_landmarks:
        # Predict the gesture based on hand landmarks
        gesture = predict_gesture(hand_landmarks)
        print(f"Predicted Gesture: {gesture_names[gesture]}")
        
        # Display the gesture name on the frame
        cv2.putText(frame, gesture_names[gesture], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Optionally, display the frame (with landmarks) in a window
    if hand_landmarks:
        for i in range(0, len(hand_landmarks), 3):
            x, y = int(hand_landmarks[i] * frame.shape[1]), int(hand_landmarks[i + 1] * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
    
    cv2.imshow("Gesture Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
