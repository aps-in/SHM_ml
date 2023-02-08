#Standard libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn 
import os

# Deep learning libraries
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

# Model evaluation libraries
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Python files
from model import CNN1D, CNN1D_F
from dataset import Dataset
from train import train, evaluate




device = 'cuda' if torch.cuda.is_available() else 'cpu'

raw_data = np.load('data.npy', allow_pickle = True)
labels = np.load('labels.npy', allow_pickle = True)

print('Data shape: ', raw_data.shape)
print('Number of data points: ', raw_data.shape[0])
print('Number of channels: ', raw_data.shape[1])
print('Signal length: ', raw_data.shape[2])


#Splitting dataset in train and val set
train_x = []
train_y = []
val_x = []
val_y = []
for i in range(7):
    current_class_data = raw_data[i*20: i*20 + 20]
    current_class_labels = labels[i*20: i*160 + 160]
    idx = np.random.permutation(20)
    current_class_data = current_class_data[idx]
    current_class_labels = current_class_labels[idx]
    train_x.append(current_class_data[0: 16])
    val_x.append(current_class_data[16: ])
    train_y.append(current_class_labels[0: 16])
    val_y.append(current_class_labels[16: ])
train_x = np.array(train_x).reshape(-1, 16, 40000)
val_x = np.array(val_x).reshape(-1, 16, 40000)
train_y = np.array(train_y).reshape(-1)
val_y = np.array(val_y).reshape(-1)

#Create dataloader
trainset = Dataset(train_x, train_y)
valset = Dataset(val_x, val_y)
batch_size = 32
train_loader = data.DataLoader(dataset = trainset, batch_size = batch_size, shuffle = True)
val_loader = data.DataLoader(dataset = valset, batch_size = batch_size, shuffle = False)


#Training model 1 with 7 calss calssificaiton
#Defining model parameters
model = CNN1D_F(7).to(device).double()
criterion = nn.CrossEntropyLoss()
learning_rate = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones = [5, 10, 15], gamma = 0.5)

num_epochs = 25
loss_train, loss_val, acc_train, acc_val = train(model, num_epochs, criterion, \
                                                        train_loader, val_loader, optimizer, scheduler, True)


#Checking perfroamnce on validaiton set for the best model
val_loader = data.DataLoader(dataset = valset, batch_size = 1, shuffle = False)

dir_name = "results/"
test = os.listdir(dir_name)
for item in test:
    if item.endswith(".pth"):
        PATH = os.path.join(dir_name, item)

weights = torch.load(PATH)
model.load_state_dict(weights)

observations = evaluate(model, val_loader)
predictions, y_test = observations[:, 0], observations[:, 1]
accuracy = accuracy_score(predictions, y_test)
print('Accuracy: ', accuracy)


#Plotting training stats and graphs
plt.figure(figsize=(8,8))
plt.plot(acc_train, label='Training Accuracy')
plt.plot(acc_val, label='Validation Accuracy')
plt.legend()
plt.title('Model accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.savefig('results/accuracy_1120.png')
plt.show()

plt.figure(figsize=(8,8))
plt.plot(loss_train, label='Training Loss')
plt.plot(loss_val, label='Validation Loss')
plt.legend()
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('results/loss_1120.png')
plt.show()

#Printing Confusion Matrix
conf_matrix = confusion_matrix(y_test, predictions)
df_cm = pd.DataFrame(conf_matrix, index = [i for i in "0123456"], columns = [i for i in "0123456"])
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('results/conf_matrix_1120.png')
plt.show()


#Load processed data and augment it
data = np.load('data_processed.npy', allow_pickle = True)
aug_data = []
for i in range(140):
    current_chunk = data[i]
    for i in range(8):
        aug_data.append(current_chunk[:, i*5000:i*5000 + 5000])

aug_data = np.array(aug_data)

labels = np.array([i for i in range(7) for j in range(160)])

#check shapes
print('Augmented data shape: ', aug_data.shape)
print('Number of data points: ', aug_data.shape[0])
print('Number of channels: ', aug_data.shape[1])
print('Signal length: ', aug_data.shape[2])


#Save augmented data
np.save('data_1120.npy', aug_data)
np.save('labels_1120.npy', labels)

#Load the saved augmented data
x_data = np.load('data_1120.npy', allow_pickle = True)
y_data = np.load('labels_1120.npy', allow_pickle = True)

#Splitting augmented data 
train_x = []
train_y = []
val_x = []
val_y = []
for i in range(7):
    current_class_data = x_data[i*160: i*160 + 160]
    current_class_labels = y_data[i*160: i*160 + 160]
    idx = np.random.permutation(160)
    current_class_data = current_class_data[idx]
    current_class_labels = current_class_labels[idx]
    train_x.append(current_class_data[0: 128])
    val_x.append(current_class_data[128: ])
    train_y.append(current_class_labels[0: 128])
    val_y.append(current_class_labels[128: ])
train_x = np.array(train_x).reshape(-1, 16, 5000)
val_x = np.array(val_x).reshape(-1, 16, 5000)
train_y = np.array(train_y).reshape(-1)
val_y = np.array(val_y).reshape(-1)



#Dataloader for augmented data
trainset = Dataset(train_x, train_y)
valset = Dataset(val_x, val_y)
batch_size = 32
train_loader = data.DataLoader(dataset = trainset, batch_size = batch_size, shuffle = True)
val_loader = data.DataLoader(dataset = valset, batch_size = batch_size, shuffle = False)

#Defining and training new model
model = CNN1D(7).to(device).double()
criterion = nn.CrossEntropyLoss()
learning_rate = 0.0005
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones = [5, 10, 15], gamma = 0.5)

num_epochs = 25
loss_train, loss_val, acc_train, acc_val = train(model, num_epochs, criterion, \
                                                        train_loader, val_loader, optimizer, scheduler, True)


#Checking performance on validation set
val_loader = data.DataLoader(dataset = valset, batch_size = 1, shuffle = False)

dir_name = "results/"
test = os.listdir(dir_name)
for item in test:
    if item.endswith(".pth"):
        PATH = os.path.join(dir_name, item)

weights = torch.load(PATH)
model.load_state_dict(weights)

observations = evaluate(model, val_loader)
predictions, y_test = observations[:, 0], observations[:, 1]
accuracy = accuracy_score(predictions, y_test)
print('Accuracy: ', accuracy)

#Training statistics 
plt.figure(figsize=(8,8))
plt.plot(acc_train, label='Training Accuracy')
plt.plot(acc_val, label='Validation Accuracy')
plt.legend()
plt.title('Model accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.savefig('results/accuracy_1120.png')
plt.show()

plt.figure(figsize=(8,8))
plt.plot(loss_train, label='Training Loss')
plt.plot(loss_val, label='Validation Loss')
plt.legend()
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('results/loss_1120.png')
plt.show()


#Printing new confusion matrix
conf_matrix = confusion_matrix(y_test, predictions)
df_cm = pd.DataFrame(conf_matrix, index = [i for i in "0123456"], columns = [i for i in "0123456"])
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('results/conf_matrix_1120.png')
plt.show()

#Dropping vlass 4 and relabel data
idx = (y_data != 4)
x_data = x_data[idx]
y_data = np.array([i for i in range(6) for j in range(160)])


#Splitting the newly formed dataset
train_x = []
train_y = []
val_x = []
val_y = []
for i in range(6):
    current_class_data = x_data[i*160: i*160 + 160]
    current_class_labels = y_data[i*160: i*160 + 160]
    idx = np.random.permutation(160)
    current_class_data = current_class_data[idx]
    current_class_labels = current_class_labels[idx]
    train_x.append(current_class_data[0: 128])
    val_x.append(current_class_data[128: ])
    train_y.append(current_class_labels[0: 128])
    val_y.append(current_class_labels[128: ])
train_x = np.array(train_x).reshape(-1, 16, 5000)
val_x = np.array(val_x).reshape(-1, 16, 5000)
train_y = np.array(train_y).reshape(-1)
val_y = np.array(val_y).reshape(-1)


#Creating dataloader

trainset = Dataset(train_x, train_y)
valset = Dataset(val_x, val_y)
batch_size = 32
train_loader = data.DataLoader(dataset = trainset, batch_size = batch_size, shuffle = True)
val_loader = data.DataLoader(dataset = valset, batch_size = batch_size, shuffle = False)

#Train new model with the class 4 removed
model = CNN1D(6).to(device).double()
criterion = nn.CrossEntropyLoss()
learning_rate = 0.0005
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones = [5, 10, 15], gamma = 0.5)

num_epochs = 25
loss_train, loss_val, acc_train, acc_val = train(model, num_epochs, criterion, \
                                                        train_loader, val_loader, optimizer, scheduler, True)

#Training graphs and confusion matrix 
val_loader = data.DataLoader(dataset = valset, batch_size = 1, shuffle = False)

dir_name = "results/"
test = os.listdir(dir_name)
for item in test:
    if item.endswith(".pth"):
        PATH = os.path.join(dir_name, item)

weights = torch.load(PATH)
model.load_state_dict(weights)

observations = evaluate(model, val_loader)
predictions, y_test = observations[:, 0], observations[:, 1]
accuracy = accuracy_score(predictions, y_test)
print('Accuracy: ', accuracy)

plt.figure(figsize=(8,8))
plt.plot(acc_train, label='Training Accuracy')
plt.plot(acc_val, label='Validation Accuracy')
plt.legend()
plt.title('Model accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.savefig('results/accuracy_1120.png')
plt.show()

plt.figure(figsize=(8,8))
plt.plot(loss_train, label='Training Loss')
plt.plot(loss_val, label='Validation Loss')
plt.legend()
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('results/loss_1120.png')
plt.show()

conf_matrix = confusion_matrix(y_test, predictions)
df_cm = pd.DataFrame(conf_matrix, index = [i for i in "012356"], columns = [i for i in "012356"])
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('results/conf_matrix_1120.png')
plt.show()

## Comparision with 5 fold accuracy
num_bars = np.arange(2)
algorithms = ['Full dataset', 'Dataset without class 4']
fig = plt.figure(figsize = (10, 8))
plt.bar(num_bars - 0.2, [83.71, 96.66], color ='r', width = 0.4, label = 'Random Forest')
plt.bar(num_bars + 0.2, [87.19, 98.21], color ='y', width = 0.4, label = '1D CNN')
plt.legend(fontsize = 12)
plt.xlabel("Dataset configuration", fontsize = 18)
plt.ylabel("5 fold accuracy", fontsize = 18)
plt.title("Comparison between ML model and DL model", fontsize = 18)
plt.xticks([i for i in range(len(algorithms))], algorithms, fontsize = 14)
plt.yticks(fontsize = 14)
plt.ylim([75, 100])
plt.show()


