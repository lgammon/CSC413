import mediapipe as mp
import cv2

# Initialize MediaPipe Objectron and Drawing Utilities
mp_objectron = mp.solutions.objectron
mp_drawing = mp.solutions.drawing_utils

# Define the object categories to detect
# Available categories: 'Shoe', 'Chair', 'Cup', 'Camera'
categories = ['Onion', 'Tomato', 'Spinach', 'Egg']

# Initialize Objectron
objectron = mp_objectron.Objectron(
    static_image_mode=False,
    max_num_objects=5,  # Detect up to 5 objects
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_name=categories[2]  # Change this to detect a specific object (e.g., 'Shoe', 'Chair')
)

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera not detected.")
    exit()

print(f"Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Flip the frame for natural mirroring
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB as MediaPipe works with RGB images
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame using Objectron
    results = objectron.process(rgb_frame)

    # Visualize the results
    if results.detected_objects:
        for detected_object in results.detected_objects:
            # Draw landmarks and edges of the detected object
            mp_drawing.draw_landmarks(
                frame,
                detected_object.landmarks_2d,
                mp_objectron.BOX_CONNECTIONS
            )
            mp_drawing.draw_axis(frame, detected_object.rotation, detected_object.translation)

            # Display the object type
            object_type = detected_object.class_name
            cv2.putText(frame, f"Object: {object_type}", 
                        (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.8, (0, 255, 0), 2)

    # Display the output frame
    cv2.imshow("Object Recognition and Classification", frame)

    # Exit when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
objectron.close()
