import torch
from torch.nn import Module, Linear, ReLU, Sigmoid
from torch.nn.init import kaiming_uniform_, xavier_uniform_
from sklearn.metrics import roc_auc_score
import numpy as np

from src.dataset import prepare_train_dataset

import os


class FraudDetectionMLP(Module):
    def __init__(self, n_inputs):
        super(FraudDetectionMLP, self).__init__()
        # Input layer
        self.hidden1 = Linear(n_inputs, 57)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        # self.dropout1 = Dropout(0.1)
        # Second (hidden) layer
        self.hidden2 = Linear(57, 22)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        # Third (hidden) layer
        self.hidden3 = Linear(22, 107)
        kaiming_uniform_(self.hidden3.weight, nonlinearity='relu')
        self.act3 = ReLU()
        # Fourth (hidden) layer
        self.hidden4 = Linear(107, 202)
        kaiming_uniform_(self.hidden4.weight, nonlinearity='relu')
        self.act4 = Sigmoid()
        # Fifth (hidden) layer
        self.hidden5 = Linear(202, 162)
        kaiming_uniform_(self.hidden5.weight, nonlinearity='relu')
        self.act5 = Sigmoid()
        # Output layer
        self.hidden6 = Linear(162,1)
        xavier_uniform_(self.hidden6.weight)
        self.act6 = Sigmoid()

    def forward(self, X):
        # Input to the first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
        # X = self.dropout1(X)
        # Second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # Third hidden layer
        X = self.hidden3(X)
        X = self.act3(X)
        # Fourth hidden layer
        X = self.hidden4(X)
        X = self.act4(X)
        # Fifth hidden layer
        X = self.hidden5(X)
        X = self.act5(X)
        # Output layer
        X = self.hidden6(X)
        X = self.act6(X)
        return X
    
def train_model(model, train_dl, num_epochs):
    # Define loss function and optimizer inside the function
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_dl:
            optimizer.zero_grad()
            outputs = model(inputs.float())
            labels = labels.float()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_dl.dataset)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1} loss: {epoch_loss:.4f}")
            
def evaluate_model(model, test_dl):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in test_dl:
            outputs = model(inputs.float())
            y_true.extend(labels.numpy())
            y_pred.extend(outputs.numpy().flatten())
    auc = roc_auc_score(y_true, y_pred)
    return auc



def run_training_evaluations(model, train_dl, test_dl):
    auc_values = []

    # Perform 10 training runs
    for i in range(10):
        # model = FraudDetectionMLP(num_features)
        train_model(model, train_dl, num_epochs=20)

        # Evaluate the model
        model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for inputs, labels in test_dl:
                outputs = model(inputs.float())
                y_true.extend(labels.numpy())
                y_pred.extend(outputs.numpy().flatten())
        auc = roc_auc_score(y_true, y_pred)
        auc_values.append(auc)  # Append AUC to the list
        print(f"Run {i+1}: AUC = {auc:.4f}")

    # Calculate the average AUC
    average_auc = np.mean(auc_values)
    print(f"\nAverage AUC: {average_auc:.4f}")

    # Calculate the std AUC
    auc_std_dev = np.std(auc_values)
    print(f"Standard Deviation of AUC: {auc_std_dev:.4f}")

    metadata = {
    'auc_values': auc_values,
    'average_auc': average_auc,
    'auc_std_dev': auc_std_dev
}

    return metadata

