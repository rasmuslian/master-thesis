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
from utils import create_folder, clear_and_create_folder
import os


"""--- DATASET ---"""

class StockGraphDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = [img for img in os.listdir(data_dir) if img.endswith('.png')]  # Adjust the extension as needed
        self.percentage_return = [float(img_name.split('__')[2]) for img_name in self.images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        image = Image.open(os.path.join(self.data_dir, img_name)).convert('RGB')
        
        # Extracting the percentage return from the filename
        percentage_return = self.percentage_return[idx]

        if self.transform:
            image = self.transform(image)

        # Convert the percentage_return to FloatTensor
        percentage_return = torch.tensor(percentage_return, dtype=torch.float32)

        return image, percentage_return

# Transforms images into same size
transform = transforms.Compose([
    transforms.Resize((64, 128)),
    transforms.ToTensor(),
])

train_folder = f"stock_graphs/{setup.train_tickerslist}/train/"
valid_folder = f"stock_graphs/{setup.train_tickerslist}/validate/"
test_folder = f"stock_graphs/{setup.test_tickerslist}/test/"

train_dataset = StockGraphDataset(train_folder, transform)
val_dataset = StockGraphDataset(valid_folder, transform)
test_dataset = StockGraphDataset(test_folder, transform)

batch_size = setup.batch_size

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # shuffle=True is only used for training dataset
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


"""--- MODEL ---"""  

pretrained_model_name = setup.pretrained_model_name
class StockGraphRegressor(nn.Module):
    def __init__(self):
        super(StockGraphRegressor, self).__init__()
        # Where we define all the parts of the model
        
        self.base_model = timm.create_model(pretrained_model_name, pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

        # enet_out_size = 1280

        match setup.pretrained_model_name:
            case 'efficientnet_b0':
                enet_out_size = 1280
            case 'efficientnet_b1':
                enet_out_size = 1280
            case 'efficientnet_b2':
                enet_out_size = 1408
            case 'efficientnet_b3':
                enet_out_size = 1536
            case 'efficientnet_b4':
                enet_out_size = 1792
            case 'efficientnet_b5':
                enet_out_size = 2048
            case 'efficientnet_b6':
                enet_out_size = 2304
            case 'efficientnet_b7' | 'tf_efficientnet_b7_ns':
                enet_out_size = 2560          

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(enet_out_size, 1)
        )

    def forward(self, x):
        x = self.features(x)
        output = self.regressor(x)
        return output

num_epochs = setup.max_epochs
train_losses, validate_losses = [], []

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = StockGraphRegressor()
model.to(device)

# Loss function
criterion = nn.MSELoss()
# Optimizer
learning_rate = setup.learning_rate
optimizer = optim.Adam(model.parameters(), lr=learning_rate) # lr = learning rate

def train_the_model(resume):
    global train_losses, validate_losses
    start_epoch = 1

    """--- RESUME TRAINING IF SET ---"""
    if resume:
        checkpoint = torch.load(f"model_training_checkpoints/{resume}.pt")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        train_losses = checkpoint['train_loss']
        validate_losses = checkpoint['val_loss']
        print(f"Resuming training from epoch {start_epoch}")
    else:
        clear_and_create_folder('model_training_checkpoints')

    """--- SAVES MODEL ---"""
    def get_model_name(actual_epochs):
        # Save todays date in a variable, in the format of jan01-1312, lowercase
        today = pd.Timestamp.today().strftime("%b%d-%H%M").lower()
        # Get 6 letter random string
        random_string = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=6))
        return f"model_{random_string}__actep{actual_epochs}_{today}_{setup.train_tickerslist}_{setup.data_interval}_{pretrained_model_name}_maxep{num_epochs}_bs{batch_size}_lr{learning_rate}".replace('.', '')
    
    def save_model(model, actual_epochs):
        create_folder('models')
        model_name = get_model_name(actual_epochs)
        torch.save(model.state_dict(), f"models/{model_name}.pt")
        print(f"Model saved as {model_name}.pt")
    
    def save_model_training_checkpoint(EPOCH, TRAIN_LOSS, VAL_LOSS):
        model_checkpoint_name = f"{pd.Timestamp.today().strftime('%b%d-%H%M').lower()}-ep{EPOCH}"
        torch.save({
            'epoch': EPOCH,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': TRAIN_LOSS,
            'val_loss': VAL_LOSS,
            }, f"model_training_checkpoints/{model_checkpoint_name}.pt" )
        print(f"Checkpoint saved as {model_checkpoint_name}.pt")

    for epoch in range(start_epoch, num_epochs + 1):
        # Training phase
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc='Training loop'):
            # Move inputs and labels to the device
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
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
                loss = criterion(outputs, labels.unsqueeze(1))
                running_loss += loss.item()
        val_loss = running_loss / len(val_loader.dataset)
        validate_losses.append(val_loss)
        print(f"Epoch {epoch}/{num_epochs} - Train loss: {train_loss}, Validation loss: {val_loss}")

        # If the validation loss is lower than the previous lowest, save the model
        if epoch == 1 or val_loss < min(validate_losses[:-1]):
            save_model(model, epoch)
        
        save_model_training_checkpoint(epoch, train_losses, validate_losses)


    # Visualize the loss over epochs
    plt.plot(train_losses, label='Training loss')
    plt.plot(validate_losses, label='Validation loss')
    plt.legend()
    plt.title("Loss over epochs")
    plt.show()


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
        output = model(image_tensor)
        predicted_value = output.item()  # Converts the tensor to a Python scalar
    return predicted_value / 100  # Convert the percentage to decimal form

# Test all test-images
def test_model():
    # Load the model
    model.load_state_dict(torch.load(f"models/{setup.test_model_name}.pt"))
    model.eval()

    error = 0
    cumulative_return = 0

    # for image_path, percentage_return in test_dataset:
    for image_path, percentage_return in zip(test_dataset.images, test_dataset.percentage_return):
        original_image, image_tensor = preprocess_image(f"stock_graphs/{setup.test_tickerslist}/test/{image_path}", transform)
        predicted_value = predict(model, image_tensor, device) * 100
        print(f"Input: {image_path}, Actual: {percentage_return}, Predicted: {predicted_value}")
        error += abs(percentage_return - predicted_value)
        cumulative_return += percentage_return
    
    average_return = cumulative_return / len(test_dataset)

    # Calculate the average error
    mean_error = error / len(test_dataset)
    mse = mean_error ** 2
    rmse = mse ** 0.5

    # Calculate 'baseline' errors
    baseline_error = 0
    for image_path, percentage_return in zip(test_dataset.images, test_dataset.percentage_return):
        baseline_error += abs(percentage_return - average_return)

    baseline_mean_error = baseline_error / len(test_dataset)
    baseline_mse = baseline_mean_error ** 2
    baseline_rmse = baseline_mse ** 0.5
        

    print(f"Average return: {'{:.2f}'.format(average_return)}%")
    return rmse, baseline_rmse

def predict_graph(graph_image_path):
    # Load the model
    model.load_state_dict(torch.load(f"models/{setup.test_model_name}.pt"))
    model.eval()
    
    original_image, image_tensor = preprocess_image(graph_image_path, transform)
    predicted_value = predict(model, image_tensor, device)

    return predicted_value
