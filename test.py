import mediapipe as mp
import cv2
import csv

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera not detected.")
    exit()

# Open CSV file for saving landmarks
with open('hand_landmarks.csv', 'w', newline='') as f:
    csv_writer = csv.writer(f)
    # Write header row
    csv_writer.writerow([f'landmark_{i}_{dim}' for i in range(21) for dim in ['x', 'y', 'z']] + ['label'])

    label = None  # Default label
    print("Press '1', '2', etc. to assign gesture labels. Press 'Esc' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Convert frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # Process landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on frame
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Save landmarks
                if label is not None:
                    row = []
                    for lm in hand_landmarks.landmark:
                        row.extend([lm.x, lm.y, lm.z])
                    row.append(label)
                    csv_writer.writerow(row)
                    print(f"Saved gesture '{label}'.")

        # Display the frame
        cv2.imshow('Hand Tracking', frame)

        # Key controls
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Esc key to exit
            break
        elif key in range(48, 58):  # Keys '0'-'9'
            label = chr(key)  # Assign label as '0', '1', ..., '9'
            print(f"Label set to: {label}")

cap.release()
cv2.destroyAllWindows()
