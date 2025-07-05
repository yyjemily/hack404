import torch 
import os 
import torch.nn as nn 
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
import torchvision.models as models
import torch.optim as optim 

from torch.utils.data import Dataset, DataLoader, random_split 
from torchvision.models import DenseNet121_Weights, densenet121
from torchvision import transforms
from torchvision.transforms import ToTensor
from PIL import Image 
from collections import Counter


#get data from dataset 
class DataSet(Dataset):
    def __init__(self, csv_file, class_list, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.class_list = class_list

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        image = Image.open(self.df.file_path[index])
        label = self.class_list.index(self.df.label[index])
  
        if self.transform:
            image = self.transform(image)
        return image, label


dataset_obj = DataSet("/Users/emiliemui/hack404/opg_classification.csv", ["BDC-BDR", "Caries", "Fractured Teeth", "Healthy Teeth", "Impacted teeth", "Infection"], transform=transforms)

def load_data(): 
    transforms = transforms.compose([
        transforms.Resize((224,224)), 
        transforms.ToTensor(), 
    ])

    dataset_size = dataset_obj.__len__()

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

    train_dataset, val_dataset, test_dataset = random_split(dataset_obj, [train_size, val_size, test_size])

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle =True) 
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle =False) 
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle =False) 
    return train_loader, val_loader, test_loader

model = models.densenet121(pretrained=True)
#get num of output features from previous layer - used for num of input features in classifier layer
input_features = model.classifier.in_features
model.classifier = nn.Linear(input_features, len(dataset_obj.class_list)) #fine tune final FC layer 

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
