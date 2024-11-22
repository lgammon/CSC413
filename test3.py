#define model
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

# Load the dataset
data = np.genfromtxt('hand_landmarks.csv', delimiter=',', skip_header=1)

# Separate features and labels
X = data[:, :-1]  # All columns except the last (63 landmarks: x, y, z)
y = data[:, -1]   # Last column (labels)

# Encode labels as integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# Create DataLoader for batching
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define the model
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

input_size = X_train.shape[1]  # Number of landmarks (63 for x, y, z)
hidden_size = 128
num_classes = len(label_encoder.classes_)
model = GestureModel(input_size, hidden_size, num_classes)

print("Model initialized successfully!")
