import numpy as np
import torch
import csv 
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def validate(model, val_loader, criterion): 
    model.eval()
    val_loss = []
    total_predictions = 0.0
    correct_predictions = 0.0
    for idx, (X,Y) in enumerate(val_loader):

        X = X.to(device)
        Y = Y.to(device)
            
        preds = model(X)
        
        loss = criterion(preds,Y)
        
        val_loss.append(loss.item())

        _, predicted = torch.max(preds, 1)
        total_predictions += Y.size(0)
        correct_predictions += (predicted == Y).sum().item()

    acc = (correct_predictions/total_predictions)*100.0

    return np.mean(val_loss), acc


def train(model, num_epochs, criterion, train_loader, val_loader, optimizer, scheduler, verbose = True):
    loss_train = []
    loss_val = []
    acc_train = []
    acc_val = []
    prev_dev_acc = 0
    for epoch in range(num_epochs): 
        model.train()
        epoch_loss = []
        total_predictions = 0.0
        correct_predictions = 0.0
        for idx, (X, Y) in enumerate(train_loader):
            X = X.to(device)
            Y = Y.to(device)
            
            optimizer.zero_grad()
            
            preds = model(X)
            
            loss = criterion(preds, Y)

            _, predicted = torch.max(preds, 1)
            total_predictions += Y.size(0)
            correct_predictions += (predicted == Y).sum().item()
            
            loss.backward()
            
            optimizer.step()

            epoch_loss.append(loss.item())

        loss_train.append(np.mean(epoch_loss))
        acc = (correct_predictions/total_predictions)*100.0
        acc_train.append(acc)

        epoch_val_loss, epoch_val_acc = validate(model, val_loader, criterion)
        loss_val.append(epoch_val_loss)
        acc_val.append(epoch_val_acc)
        if epoch_val_acc > prev_dev_acc:
            
            dir_name = "results/"
            test = os.listdir(dir_name)

            for item in test:
                if item.endswith(".pth"):
                    os.remove(os.path.join(dir_name, item))
            
            path = 'results/model_parameters_' + str(epoch_val_acc)[0: 5] + '.pth'
            torch.save(model.state_dict(), path)
            print('Saving model parameters...')
            print('Validation accuracy: ', epoch_val_acc)
            prev_dev_acc = epoch_val_acc
        

        if scheduler != None:
            scheduler.step()

        if verbose:
            print('EPOCH: ' + str(epoch))
            print('TRAIN_LOSS: ' + str(loss_train[-1]))
            print('TRAIN_ACC: ' + str(acc_train[-1]))
            print('VAL_LOSS: ' + str(loss_val[-1]))
            print('VAL_ACC: ' + str(acc_val[-1]))
            print('+'*25)


    return loss_train, loss_val, acc_train, acc_val



def evaluate(model, loader):
    observations = []
    heading = ['Predictions', 'Labels']
    model.eval()
    for idx, (X, Y) in enumerate(loader):
        X = X.to(device)
        Y = Y.to(device)      
        preds = model(X)
        _, predicted = torch.max(preds, 1)
        observations.append([predicted.item(), Y.item()])
    
    return np.array(observations)

    
