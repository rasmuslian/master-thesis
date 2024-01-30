import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm
from PIL import Image
import random

import matplotlib.pyplot as plt # For data viz
import pandas as pd
import numpy as np
import sys
from tqdm import tqdm
import setup
from utils import create_folder


"""--- DATASET ---"""

class StockGraphDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    @property
    def classes(self):
        return self.data.classes

# Transforms images into same size
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

train_folder = 'output/train/'
valid_folder = 'output/validate/'
test_folder = 'output/test/'

train_dataset = StockGraphDataset(train_folder, transform)
val_dataset = StockGraphDataset(valid_folder, transform)
test_dataset = StockGraphDataset(test_folder, transform)

batch_size = setup.batch_size

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # shuffle=True is only used for training dataset
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Get a dictionary associating target values with folder names
target_to_class = {v: k for k, v in ImageFolder(train_folder).class_to_idx.items()}
num_classes = len(target_to_class)


"""--- MODEL ---"""  

pretrained_model_name = setup.pretrained_model_name
class StockGraphClassifer(nn.Module):
    def __init__(self):
        super(StockGraphClassifer, self).__init__()
        # Where we define all the parts of the model
        
        self.base_model = timm.create_model(pretrained_model_name, pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

        enet_out_size = 1280
        # Make a classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(enet_out_size, num_classes)
        )
    
    def forward(self, x):
        # Connect these parts and return the output
        x = self.features(x)
        output = self.classifier(x)
        return output

num_epochs = setup.num_epochs
train_losses, validate_losses = [], []

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = StockGraphClassifer()
model.to(device)

# Loss function
criterion = nn.CrossEntropyLoss()
# Optimizer
learning_rate = setup.learning_rate
optimizer = optim.Adam(model.parameters(), lr=learning_rate) # lr = learning rate

def train_the_model():
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc='Training loop'):
            # Move inputs and labels to the device
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc='Validation loop'):
                # Move inputs and labels to the device
                images, labels = images.to(device), labels.to(device)
            
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * labels.size(0)
        val_loss = running_loss / len(val_loader.dataset)
        validate_losses.append(val_loss)
        print(f"Epoch {epoch+1}/{num_epochs} - Train loss: {train_loss}, Validation loss: {val_loss}")

    # Visualize the loss over epochs
    # plt.plot(train_losses, label='Training loss')
    # plt.plot(validate_losses, label='Validation loss')
    # plt.legend()
    # plt.title("Loss over epochs")
    # plt.show()
        
    """--- SAVES MODEL ---"""
    def get_model_name():
        # Save todays date in a variable, in the format of jan01, lowercase
        today = pd.Timestamp.today().strftime("%b%d").lower()
        return f"model_{today}_{setup.training_data_name}_{pretrained_model_name}_ep{num_epochs}_bs{batch_size}_lr{learning_rate}".replace('.', '')

    create_folder('models')
    model_name = get_model_name()
    torch.save(model.state_dict(), f"models/{model_name}.pt")
    print(f"Model saved as {model_name}.pt")


"""--- EVALUATION ---"""

# Load and preprocess the image
def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    return image, transform(image).unsqueeze(0)

# Predict using the model
def predict(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return probabilities.cpu().numpy().flatten()

# Test all test-images
def test_model():
    # Load the model
    model.load_state_dict(torch.load(f"models/{setup.test_model_name}.pt"))
    model.eval()

    score = 0
    for image_path, label in train_dataset.data.imgs:
        original_image, image_tensor = preprocess_image(image_path, transform)
        probabilities = predict(model, image_tensor, device)
        predicted_class = target_to_class[np.argmax(probabilities)]
        print(f"Input: {image_path}, Label: {label}, Predicted: {predicted_class}, Probabilities: {probabilities}")
        if predicted_class == target_to_class[label]:
            score += 1
    
    # Gets random result benchmark
    rand_score = 0
    for image_path, label in train_dataset.data.imgs:
        rand_score += 1 if random.random() < 0.5 else 0

    score = score / len(train_dataset)
    rand_score = rand_score / len(train_dataset)
    return score, rand_score