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

from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset, DataLoader, random_split 
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

#WEIGHT SAMPLING STUFFFF ADDDDD LATERRRRRRR--------------------------------------
# #make class weights 

# class_count = Counter() #stores amt of data in each classification
# for ind in len(dataset_obj.class_list):
#     class_count[ind] += 1

# class_weights = {}
# #calculate class weights 
# for class_ind in class_count:
#     count = class_count[class_ind]
#     class_weights[class_ind] = (dataset_obj.__len__)/count


#split data 80-10-10
train_size = int(0.8 * dataset_size)
val_size = int(0.10 * dataset_size)
test_size = dataset_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle =True) 
val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle =False) 
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle =False) 

print(train_loader)
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

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(trainable_params, lr=1e-3)


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_acc = 0.0
    
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
        
        with torch.no_grad(): 
            for images, labels in val_loader:
                labels = labels.long()  
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct_val / total_val
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print('-' * 50)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_dental_model.pth')
            print(f'New best model saved! Val Acc: {val_acc:.2f}%')
    
    return train_losses, val_losses, train_accuracies, val_accuracies

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
    print(f'Test Accuracy: {test_acc:.2f}%')
    
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
    
    return test_acc

# Plot training history
def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
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
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Starting training...")
    print(f"Number of classes: {len(class_list)}")
    print(f"Classes: {class_list}")
    
    # Train the model
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs=25
    )
    
    # Plot training history
    #plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies)
    
    # Load best model and test
    model.load_state_dict(torch.load('new_dental_model.pth'))
    test_accuracy = test_model(model, test_loader)
    
    print(f"\nTraining completed! Best model saved as 'dental_model.pth'")
    print(f"Final test accuracy: {test_accuracy:.2f}%")