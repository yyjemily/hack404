import torch 
import os 
import torch.nn as nn 
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
import torchvision.models as models
import torch.optim as optim 
import seaborn as sns
import torchvision 
from collections import Counter

from sklearn.metrics import confusion_matrix, classification_report, f1_score
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torchvision.models import DenseNet121_Weights, densenet121
from torchvision import transforms
from torchvision.transforms import ToTensor
from PIL import Image, ImageFilter

torch.manual_seed(1) 

# #get data from dataset 
# class DataSet(Dataset):
#     def __init__(self, csv_file, class_list, transform=None):
#         self.df = pd.read_csv(csv_file)
#         self.transform = transform
#         self.class_list = class_list

#     def __len__(self):
#         return self.df.shape[0]

#     def __getitem__(self, index):
#         image = Image.open(self.df.iloc[index]['/opg_classification.csv'])
#         label = self.class_list.index(self.df.iloc[index]['label'])
  
#         if self.transform:
#             image = self.transform(image)
#         return image, label


# dataset = DataSet("/opg_classification.csv", ["BDC-BDR, Caries", "Fractured Teeth", "Healthy Teeth", "Impacted teeth", "Infection"], transform=transforms)
class_list = ["BDC-BDR", "Caries", "Fractured Teeth", "Healthy Teeth", "Impacted teeth", "Infection"]

#preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# augmentation 
augment_transform = transforms.Compose([
   transforms.Resize((224, 224)),
   transforms.RandomHorizontalFlip(p=1.0),
   transforms.Lambda(lambda img: img.filter(ImageFilter.GaussianBlur(radius=1))),
   transforms.ToTensor(),
   transforms.Lambda(lambda x: x + 0.05 * torch.randn_like(x)),  #add noise, req a tensor 
   transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

dataset = torchvision.datasets.ImageFolder(
    root="/Users/emiliemui/Downloads/Dental OPG XRAY Dataset/Dental OPG (Classification)", 
    transform=transform
)
dataset_size = len(dataset)

# Calculate class weights to handle imbalanced data
def calculate_class_weights(dataset):
    """Calculate class weights for imbalanced dataset"""
    class_counts = Counter()
    for _, label in dataset:
        class_counts[label] += 1
    
    total_samples = len(dataset)
    class_weights = {}
    
    for class_idx in range(len(class_list)):
        if class_idx in class_counts:
            # Inverse frequency weighting: more samples = lower weight
            class_weights[class_idx] = total_samples / (len(class_list) * class_counts[class_idx])
        else:
            class_weights[class_idx] = 1.0
    
    # Convert to tensor
    weights_tensor = torch.FloatTensor([class_weights[i] for i in range(len(class_list))])
    
    print("Class distribution:")
    for i, class_name in enumerate(class_list):
        count = class_counts.get(i, 0)
        weight = class_weights[i]
        print(f"{class_name}: {count} samples, weight: {weight:.3f}")
    
    return weights_tensor

# Calculate class weights
class_weights = calculate_class_weights(dataset)

#split data 80-10-10
train_size = int(0.8 * dataset_size)
val_size = int(0.10 * dataset_size)
test_size = dataset_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Calculate sample weights for weighted sampling
def get_sample_weights(dataset_split):
    """Get sample weights for weighted random sampling"""
    sample_weights = []
    for idx in dataset_split.indices:
        _, label = dataset[idx]
        sample_weights.append(class_weights[label].item())
    return sample_weights

# Get sample weights for training set
train_sample_weights = get_sample_weights(train_dataset)
train_sampler = WeightedRandomSampler(
    weights=train_sample_weights,
    num_samples=len(train_sample_weights),
    replacement=True
)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler) 
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) 
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) 

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")

model = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
#get num of output features from previous layer - used for num of input features in classifier layer
input_features = model.classifier.in_features
model.classifier = nn.Linear(input_features, len(class_list)) #fine tune final FC layer 

#freeze everything but last layer for fine tuning

for param in model.parameters():
    param.requires_grad = False

for param in model.classifier.parameters():
    param.requires_grad = True 

#get trainable parameters 
trainable_params = []
for param in model.parameters():
    if param.requires_grad:
        trainable_params.append(param)

# Use weighted CrossEntropyLoss to handle class imbalance
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(trainable_params, lr=1e-3)


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    val_f1_scores = []
    
    best_val_acc = 0.0
    best_val_f1 = 0.0
    patience = 5
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for (image, label) in train_loader: 
            actual = label.long() 

            #forward pass through model 
            out = model(image)
            loss = criterion(out, actual)

            #backward
            optimizer.zero_grad() 
            loss.backward()
            optimizer.step() 

            # Calculate training metrics
            running_loss += loss.item()
            _, predicted = torch.max(out.data, 1)
            total_train += actual.size(0)
            correct_train += (predicted == actual).sum().item()
        
        # Calculate training metrics
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        all_val_preds = []
        all_val_labels = []
        
        with torch.no_grad(): 
            for images, labels in val_loader:
                labels = labels.long()  
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct_val / total_val
        
        # Calculate F1 score for validation
        val_f1 = f1_score(all_val_labels, all_val_preds, average='weighted')
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        val_f1_scores.append(val_f1)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}')
        print('-' * 50)
        
        # Early stopping based on F1 score (better for imbalanced data)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_dental_model.pth')
            print(f'New best model saved! Val F1: {val_f1:.4f}, Val Acc: {val_acc:.2f}%')
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
    
    return train_losses, val_losses, train_accuracies, val_accuracies, val_f1_scores

# Test function
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_acc = 100 * correct / total
    test_f1 = f1_score(all_labels, all_predictions, average='weighted')
    
    print(f'Test Accuracy: {test_acc:.2f}%')
    print(f'Test F1 Score: {test_f1:.4f}')
    
    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=class_list))
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_list, yticklabels=class_list)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    return test_acc, test_f1

# Plot training history
def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies, val_f1_scores):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot losses
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(train_accuracies, label='Training Accuracy')
    ax2.plot(val_accuracies, label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    # Plot F1 scores
    ax3.plot(val_f1_scores, label='Validation F1 Score', color='red')
    ax3.set_title('Validation F1 Score')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('F1 Score')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Starting training...")
    print(f"Number of classes: {len(class_list)}")
    print(f"Classes: {class_list}")
    
    # Train the model
    train_losses, val_losses, train_accuracies, val_accuracies, val_f1_scores = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs=25
    )
    
    # Plot training history
    plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies, val_f1_scores)
    
    # Load best model and test
    model.load_state_dict(torch.load('best_dental_model.pth'))
    test_accuracy, test_f1 = test_model(model, test_loader)
    
    print(f"\nTraining completed! Best model saved as 'best_dental_model.pth'")
    print(f"Final test accuracy: {test_accuracy:.2f}%")
    print(f"Final test F1 score: {test_f1:.4f}")