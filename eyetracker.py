import cv2
import numpy as np
import matplotlib
import time
matplotlib.use('TkAgg')  # Use a GUI backend
import matplotlib.pyplot as plt
from collections import deque

# Load Haar cascade for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Initialize variables to store pupil coordinates
pupil_positions = deque()  # Store all positions

# Start capturing video
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Set timer (5 minutes and 5 seconds = 305 seconds)
start_time = time.time()
duration = 305  # seconds

while cap.isOpened():
    elapsed_time = time.time() - start_time
    if elapsed_time > duration:
        print("Time limit reached. Stopping recording.")
        break
    
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # Improve contrast for detection
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(50, 50))
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
        
        # Expand detection to include areas slightly below the detected eyes
        expanded_eyes = []
        for (ex, ey, ew, eh) in eyes:
            expanded_eyes.append((ex, ey, ew, eh))
            expanded_eyes.append((ex, ey + int(eh * 0.3), ew, eh))  # Add a lower eye region
        
        # Limit detection to only two eyes based on highest y-coordinates (last 2 in sorted list)
        eyes = sorted(expanded_eyes, key=lambda e: e[1])[-2:]
        
        for (ex, ey, ew, eh) in eyes:
            eye_roi = roi_gray[ey:ey+eh, ex:ex+ew]
            eye_roi = cv2.GaussianBlur(eye_roi, (5, 5), 0)  # Reduce noise
            threshold = cv2.adaptiveThreshold(eye_roi, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 4)
            
            contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest_contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00']) + x + ex
                    cy = int(M['m01'] / M['m00']) + y + ey
                    pupil_positions.append((cx, cy))
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
    
    cv2.imshow('Eye Tracking', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("User stopped recording.")
        break

cv2.waitKey(1000)  # Wait 1 second before closing
cap.release()
cv2.destroyAllWindows()

# Convert pupil positions to numpy arrays for plotting
pupil_positions = np.array(pupil_positions)

print(f"Number of tracked points: {len(pupil_positions)}")

if len(pupil_positions) > 0:
    plt.figure(figsize=(8, 6))
    plt.scatter(pupil_positions[:, 0], pupil_positions[:, 1], c=np.arange(len(pupil_positions)), cmap='rainbow', s=10)
    plt.gca().invert_yaxis()
    plt.colorbar(label='Frame Index')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Eye Movement Tracking')
    
    # Save the plot as a JPEG file
    plt.savefig("eye_movement.jpg", dpi=300)
    print("Plot saved as eye_movement.jpg")
    
    plt.show(block=True)
