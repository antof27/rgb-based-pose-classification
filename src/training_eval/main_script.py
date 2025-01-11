import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from torchvision import models
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_fscore_support, confusion_matrix
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
import json
from models_initialization import initialize_model_efficientnet, initialize_model_resnet, initialize_model_mobilenet, initialize_model_resnext, initialize_model_inceptionv3
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import random
import skopt
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
import sys 


#Usage
#python3 model.py --weights non-pretrained --mode normal --n_gpu 1
#python3 model.py --weights non-pretrained --mode depth --n_gpu 1
#python3 model.py --weights pretrained --mode normal --n_gpu 1
#python3 model.py --weights pretrained --mode depth --n_gpu 2

MODEL = 'Inceptionv3'
MODE = 'depth'
N_GPU = 2
WEIGHTS = "pretrained"
BATCH_SIZE = 64

# Argument parser
for i in range(1, len(sys.argv), 2):
    if sys.argv[i] == "--weights":
        WEIGHTS = sys.argv[i + 1]
    elif sys.argv[i] == "--mode":
        MODE = sys.argv[i + 1]
    elif sys.argv[i] == "--n_gpu":
        N_GPU = int(sys.argv[i + 1])

best_weights = None


# Retrieve the dataset
class CustomImageDataset(Dataset):
    def __init__(self, csv_file, image_folder, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.transform = transform
        self.data.iloc[:, 1:] = self.data.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').fillna(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename = self.data.iloc[idx, 0]
        img_path = os.path.join(self.image_folder, filename + '.jpg')
        image = Image.open(img_path).convert("RGB")
        label = self.data.iloc[idx, 1:].values
        label = np.array(label, dtype=np.float32)
        class_index = np.argmax(label)
        
        if self.transform:
            image = self.transform(image)

        return image, class_index

#Early stopping implementation
class EarlyStopping:
    def __init__(self, patience=15, threshold=0.001):
        self.patience = patience
        self.threshold = threshold
        self.best_score = None
        self.counter = 0

    def step(self, score):
        if self.best_score is None:
            self.best_score = score  
            return False

        improvement = score - self.best_score
        if improvement >= self.threshold:
            self.best_score = score
            self.counter = 0       
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Stopping early. No significant improvement for {self.patience} epochs.")
                return True
        return False

#Function to calculate the normalized confusion matrix
def normalize_confusion_matrix(conf_matrix):
    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    normalized = conf_matrix / row_sums.astype(np.float32)
    return normalized

# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss) 
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


# Initialize paths and seeds
train_csv = "/home/afinocchiaro/dm/frames_dataset/train_data.csv"
test_csv = "/home/afinocchiaro/dm/frames_dataset/test_data.csv"
image_folder = f"/home/afinocchiaro/dm/frames_dataset/{MODE}"
validation_image_folder = f"/home/afinocchiaro/dm/frames_dataset/{MODE}"

#Seeds
seed = 27
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#Dataset transformations
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(224), #set it to 299 for InceptionV3
    transforms.RandomApply([transforms.GaussianBlur(3)], p=0.5),
    transforms.ToTensor(),
])


train_dataset = CustomImageDataset(csv_file=train_csv, image_folder=image_folder, transform=transform)
test_dataset = CustomImageDataset(csv_file=test_csv, image_folder=validation_image_folder, transform=transform)

#Optimizers and activation functions pool for Bayesian Optimization
# optimizers = {
#     'adam': lambda params: optim.Adam(params, lr = 0.01),
#     #'sgd': lambda params: optim.SGD(params, momentum=0.9, weight_decay=1e-5, lr = 0.01),
#     #'adamw': lambda params: optim.AdamW(params, weight_decay=1e-5, lr = 0.01),
#     #'rmsprop': lambda params: optim.RMSprop(params, weight_decay=1e-5, lr = 0.01),
# }

# activation_functions = {
#     #'relu': nn.ReLU(),
#     #'leaky_relu': nn.LeakyReLU(),
#     #'silu': nn.SiLU(),
#     'softplus': nn.Softplus(),
# }


device = torch.device(f"cuda:{N_GPU}" if torch.cuda.is_available() else "cpu")
scaler = torch.amp.GradScaler()


def initialize_model(model_name, num_classes, activation_name):
    if model_name == "EfficientNetv2":
        return initialize_model_efficientnet(num_classes, activation_name, WEIGHTS)
    elif model_name == "ResNet101":
        return initialize_model_resnet(num_classes, activation_name, WEIGHTS)
    elif model_name == "MobileNetv3":
        print("Using MobileNet")
        return initialize_model_mobilenet(num_classes, activation_name, WEIGHTS)
    elif model_name == "ResNeXt101":
        print("Using ResNeXt")
        return initialize_model_resnext(num_classes, activation_name, WEIGHTS)
    elif model_name == "Inceptionv3":
        return initialize_model_inceptionv3(num_classes, activation_name, WEIGHTS)
    else:
        raise ValueError(f"Model {model_name} is not supported.")





def train_and_validate(model, train_dataset, test_dataset, optimizer, criterion, num_epochs, early_stopping, device, scheduler=None):
    batch_size = BATCH_SIZE
    print(f"Training with fixed batch_size: {batch_size}")


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, prefetch_factor=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, prefetch_factor=2)

     # Scheduler for learning rate reduction
    if scheduler is None:
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=num_epochs)

    best_accuracy = 0.0
    best_weights = None
    best_conf_matrix = None  # To store the best confusion matrix

    metrics = []  # Store metrics for each epoch

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type ='cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            loss = loss / 4  
            scaler.scale(loss).backward()

            # Gradient Clipping - Clip gradients by norm, performance drop
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Adjust max_norm as needed

            if (i + 1) % 4 == 0:
                scaler.step(optimizer)
                scaler.update()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                with torch.amp.autocast(device_type ='cuda'):
                    outputs = model(images)

                predicted = torch.argmax(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = correct / total if total > 0 else 0

        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
        )

        metrics.append({
            'epoch': epoch + 1,
            'training_loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        })

        print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_loss:.4f}, Validation Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

        # Check if current configuration is the best
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_weights = model.state_dict()
            best_conf_matrix = confusion_matrix(all_labels, all_preds)

            # Save metrics and confusion matrix
            with open(f"/home/afinocchiaro/dm/src/{MODEL}/{WEIGHTS}/{MODE}_images/metric_results.json", 'w') as json_file:
                json.dump(metrics, json_file, indent=4)

            with open(f"/home/afinocchiaro/dm/src/{MODEL}/{WEIGHTS}/{MODE}_images/best_confusion_matrix.json", 'w') as cm_file:
                json.dump(best_conf_matrix.tolist(), cm_file, indent=4)

        # Early stopping trigger
        if early_stopping.step(accuracy):
            print(f"Early stopping triggered at epoch {epoch + 1}. Best validation accuracy: {early_stopping.best_score:.4f}.")
            break

        # Step the scheduler
        scheduler.step()

    return best_accuracy, best_weights

#Uncomment for Bayesian Optimization
'''
# Objective function for optimization
def objective(params):
    global best_weights
    optimizer_name, activation_name = params

    # Initialize model and optimizer
    model = initialize_model(MODEL, num_classes=10, activation_name=activation_name).to(device)
    optimizer_func = optimizers[optimizer_name]
    optimizer = optimizer_func(model.parameters())

    # Create a fresh EarlyStopping instance for each run in the Bayesian optimization
    early_stopping = EarlyStopping(patience=6, threshold=0.001)

    # Scheduler for learning rate reduction
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # Adjust learning rate every 3 epochs

    # Train and validate model with early stopping
    best_loss, weights = train_and_validate(
        optimizer, model, num_epochs=50, optimizer_name=optimizer_name, activation_name=activation_name,
        early_stopping=early_stopping, scheduler=scheduler
    )

    if best_loss < float('inf'):  
        best_weights = weights

    return best_loss  # Minimize loss


Define search space for Bayesian optimization
space = [
    Categorical(['adam'], name='optimizer'),
    Categorical(['softplus'], name='activation'),
]

# Early stopping settings
early_stopping = EarlyStopping(patience=10, threshold=0.001)
'''

if __name__ == "__main__":
    activation_function = nn.Softplus()
    
    model = initialize_model(MODEL, num_classes=10, activation_name=activation_function).to(device)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=100)
    criterion = FocalLoss(alpha=1, gamma=2)
    early_stopping = EarlyStopping(patience=15, threshold=0.001)

    best_accuracy, best_weights = train_and_validate(
        model, train_dataset, test_dataset, optimizer, criterion, num_epochs=100, early_stopping=early_stopping, device=device, scheduler=scheduler)

    # Save the best model weights
    if best_weights:
        save_path = f"/home/afinocchiaro/dm/src/{MODEL}/{WEIGHTS}/{MODE}_images/final_model_weights.pth"
        torch.save(best_weights, save_path)
        print(f"Best model weights saved to {save_path}.")
    else:
        print("Training did not produce any best weights.")