"Test objects- no cam"
import cv2
import torch
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image
import torch.nn as nn
import time

# Define the categories for classification
categories = ["Egg", "Onion", "Spinach", "Tomato"]

# Load the trained model
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(categories))
model.load_state_dict(torch.load("object_classifier.pth", weights_only=True))
model.eval()

# Data transformation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Open the webcam
cap = cv2.VideoCapture(0)

# Confidence threshold
confidence_threshold = 0.8

# Duration to capture frames
capture_duration = 5  # seconds

# Track predictions
predictions = {}

start_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame for mirror effect
    frame = cv2.flip(frame, 1)

    # Preprocess the frame
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = transform(img).unsqueeze(0)

    # Predict the category
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

        if confidence.item() >= confidence_threshold:
            category = categories[predicted.item()]
            if category in predictions:
                predictions[category] += confidence.item()
            else:
                predictions[category] = confidence.item()

    # Check if 5 seconds have elapsed
    if time.time() - start_time > capture_duration:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Find the object with the highest total confidence
if predictions:
    identified_object = max(predictions, key=predictions.get)
    print(identified_object)
else:
    print("None")  # Return "None" if no confident predictions were made
