import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


# Define the model structure
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

# Load test data
data = np.genfromtxt('hand_landmarks.csv', delimiter=',', skip_header=1)
X = data[:, :-1]  # All columns except the last
y = data[:, -1]   # Last column (labels)

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing
_, X_test, _, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Convert test data to tensors
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=32)

# Reload the trained model
input_size = X_test.shape[1]
hidden_size = 128
num_classes = len(label_encoder.classes_)
model = GestureModel(input_size, hidden_size, num_classes)

# Load model weights
model.load_state_dict(torch.load('gesture_model.pth'))
model.eval()

# Evaluate the model
y_preds = []
y_true = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        y_preds.extend(predicted.numpy())
        y_true.extend(y_batch.numpy())

print(f'Accuracy: {accuracy_score(y_true, y_preds)}')
