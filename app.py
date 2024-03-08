import numpy as np
import pandas as pd
import torch
from numpy import vstack
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import Tensor
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Tanh
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD
from torch.optim import Adam
from torch.nn import BCELoss
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
import time
import copy
import streamlit as st
import matplotlib.pyplot as plt

def main():
    st.title("Fraud Detection in Financial Statements")
    
    # Filter the DataFrame for years from 1990 to 2019
    df_filtered = df[(df['fyear'] >= 1990) & (df['fyear'] <= 2019)]
    
    # Display the slider for selecting the cutoff year
    cutoff_year = st.slider("Select Cutoff Year for Training and Testing Data", min_value=1990, max_value=2019, value=2002, step=1)
    
    # Split the data based on the cutoff year
    train_data = df_filtered[df_filtered['fyear'] <= cutoff_year]
    test_data = df_filtered[df_filtered['fyear'] > cutoff_year]
    
    # plot the number of misstatements by fiscal year
    total_misstatements, total_misstatements_training, total_misstatements_testing = data_plotting(df_filtered)
    st.write(f"Total Misstatements: {total_misstatements}")
    st.write(f"Total Misstatements in Training Data: {total_misstatements_training}")
    st.write(f"Total Misstatements in Testing Data: {total_misstatements_testing}")

    # # Display some information about the datasets
    # st.write(f"Training Data: {len(train_data)} records")
    # st.write(f"Testing Data: {len(test_data)} records")


def data_ingestion():
    # Load the dataset
    url = "https://media.githubusercontent.com/media/syedshahlal/MDA_TextAnalysis/main/dataset/merged_compustat_and_labels.csv"
    df = pd.read_csv(url)
    return df

def data_preprocessing(df):
    # preprocess the dataset
    # Step 1: fill missing values with 0
    df = df.fillna(0)
    df = df[df.columns].replace([np.inf, -np.inf], 0)

    columns_to_normalize = ['act', 'ap', 'at', 'ceq', 'che', 'cogs', 'csho', 'dlc', 'dltis', 'dltt', 'dp', 'ib',
                       'invt', 'ivao', 'ivst', 'lct', 'lt', 'ni', 'ppegt', 'pstk', 're', 'rect', 'sale', 'sstk',
                       'txp', 'txt', 'xint', 'prcc_f', 'dch_wc', 'ch_rsst', 'dch_rec', 'dch_inv', 'soft_assets',
                       'ch_cs', 'ch_cm', 'ch_roa', 'bm', 'dpi', 'reoa', 'EBIT', 'ch_fcf', 'issue']
    # Step 2: Apply Min-Max scaling for normalization
    scaler = MinMaxScaler()
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

    # Step 3: Convert the normalized values to float type (should already be in float)
    df[columns_to_normalize] = df[columns_to_normalize].astype('float32')

    # Step 4: Reorder the columns
    desired_columns_order = [
    'gvkey', 'fyear', 'tic', 'cik', 'Bank', 'act', 'ap', 'at', 'ceq', 'che',
    'cogs', 'csho', 'dlc', 'dltis', 'dltt', 'dp', 'ib', 'invt', 'ivao', 'ivst',
    'lct', 'lt', 'ni', 'ppegt', 'ppent', 'pstk', 're', 'rect', 'sale', 'sstk',
    'txp', 'txt', 'xint', 'prcc_f', 'dch_wc', 'ch_rsst', 'dch_rec', 'dch_inv',
    'soft_assets', 'ch_cs', 'ch_cm', 'ch_roa', 'issue', 'bm', 'dpi', 'reoa',
    'EBIT', 'ch_fcf', 'misstate']

    df = df.reindex(columns=desired_columns_order)

    return df

def plot_misstatements(df):
    # Filter the DataFrame for years from 1990 to 2019
    filtered_df = df[(df['fyear'] >= 1990) & (df['fyear'] <= 2019)]
    training_df = df[(df['fyear'] >= 1990) & (df['fyear'] <= 2002)]
    testing_df = df[(df['fyear'] >= 2003) & (df['fyear'] <= 2019)]

    # Group by fiscal year and sum misstatements
    misstatements_by_fyear = filtered_df.groupby('fyear')['misstate'].sum()
    misstatements_by_fyear_training = training_df.groupby('fyear')['misstate'].sum()
    misstatements_by_fyear_testing = testing_df.groupby('fyear')['misstate'].sum()

    # Calculate total misstatements
    total_misstatements = misstatements_by_fyear.sum()
    total_misstatements_training = misstatements_by_fyear_training.sum()
    total_misstatements_testing = misstatements_by_fyear_testing.sum()

    # Plotting
    misstatements_by_year = df.groupby('fyear')['misstate'].sum()
    
    plt.figure(figsize=(10, 5))
    misstatements_by_year.plot(kind='bar')
    plt.axvline(x=cutoff_year-1990, color='red', linestyle='--', label='Train-Test Cutoff')  # Adjusting x to match index in plot
    plt.xlabel('Fiscal Year')
    plt.ylabel('Number of Misstatements')
    plt.title('Misstatements by Fiscal Year')
    plt.legend()
    
    st.pyplot(plt)

    return total_misstatements, total_misstatements_training, total_misstatements_testing

def data_splitting(df, train_period, test_period):
    # Split features into the specified groups
    raw_items_28_financial_ratios_14 = ['act', 'ap', 'at', 'ceq', 'che', 'cogs', 'csho', 'dlc', 'dltis', 'dltt', 'dp', 'ib',
                       'invt', 'ivao', 'ivst', 'lct', 'lt', 'ni', 'ppegt', 'pstk', 're', 'rect', 'sale', 'sstk',
                       'txp', 'txt', 'xint','prcc_f', 'dch_wc', 'ch_rsst', 'dch_rec', 'dch_inv', 'soft_assets',
                       'ch_cs', 'ch_cm', 'ch_roa', 'bm', 'dpi', 'reoa', 'EBIT', 'ch_fcf', 'issue']
    
    # Assign Train, Val, and Test periods
    # train_period, test_period = (1990, 2002), (2003, 2019)

    # loading data
    train_data = df[df['fyear'] <= cutoff_year]
    test_data = df[df['fyear'] > cutoff_year]

    # Extract features (X) and target variable (y) for training and testing
    X_train = train_data[raw_items_28_financial_ratios_14]
    y_train = train_data['misstate']

    X_test = test_data[raw_items_28_financial_ratios_14]
    y_test = test_data['misstate']

    # Undersampling the data
    rus = RandomUnderSampler()
    X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)

    return X_train_resampled, y_train_resampled, X_test, y_test

def fin_ratio(X_train_resampled, y_train_resampled, X_test, y_test):
    financial_ratios_14 = ['dch_wc', 'ch_rsst', 'dch_rec', 'dch_inv', 'soft_assets', 'ch_cs', 'ch_cm', 'ch_roa', 'bm',
                          'dpi', 'reoa', 'EBIT', 'ch_fcf', 'issue']

    raw_financial_items_28 = ['act', 'ap', 'at', 'ceq', 'che', 'cogs', 'csho', 'dlc', 'dltis', 'dltt', 'dp', 'ib',
                        'invt', 'ivao', 'ivst', 'lct', 'lt', 'ni', 'ppegt', 'pstk', 're', 'rect', 'sale', 'sstk',
                        'txp', 'txt', 'xint', 'prcc_f']

    X_train_resampled_28 = X_train_resampled.loc[:, raw_financial_items_28]
    y_train_resampled_28 = y_train_resampled

    X_test_28 = X_test.loc[:, raw_financial_items_28]
    y_test_28 = y_test

    X_train_resampled_14 = X_train_resampled.loc[:, financial_ratios_14]
    y_train_resampled_14 = y_train_resampled

    X_test_14 = X_test.loc[:, financial_ratios_14]
    y_test_14 = y_test

    merged_train_data = pd.concat([X_train_resampled, y_train_resampled], axis = 1)
    merged_test_data = pd.concat([X_test, y_test], axis = 1)

    merged_train_data_28 = pd.concat([X_train_resampled_28, y_train_resampled_28], axis = 1)
    merged_test_data_28 = pd.concat([X_test_28, y_test_28], axis = 1)

    merged_train_data_14 = pd.concat([X_train_resampled_14, y_train_resampled_14], axis = 1)
    merged_test_data_14 = pd.concat([X_test_14, y_test_14], axis = 1)

    # merged_train_data.to_csv('/dataset/merged_train_data.csv', index=False)
    # merged_test_data.to_csv('/dataset/merged_test_data.csv', index=False)

    # merged_train_data_28.to_csv('/dataset/merged_train_data_28.csv', index=False)
    # merged_test_data_28.to_csv('/dataset/merged_test_data_28.csv', index=False)

    # merged_train_data_14.to_csv('/dataset/merged_train_data_14.csv', index=False)
    # merged_test_data_14.to_csv('/dataset/merged_test_data_14.csv', index=False)

    return merged_train_data, merged_test_data, merged_train_data_28, merged_test_data_28, merged_train_data_14, merged_test_data_14

class CSVDataset(Dataset):
    #Constructor for initially loading
    def __init__(self, path):
        df = read_csv(path, header=0)
        self.X = df.values[0:, :-1]
        self.y = df.values[0:, -1]
        self.X = self.X.astype('float32')
        self.y = self.y.astype('float32')
        self.y = self.y.reshape((len(self.y), 1))

        print(self.X.shape)
        print(self.y.shape)

    # Get the number of rows in the dataset
    def __len__(self):
        return len(self.X)
    # Get a row at an index
    def __getitem__(self,idx):
        return [self.X[idx], self.y[idx]]

    def prepare_train_dataset(path):
        train = CSVDataset(path)
        train_dl = DataLoader(train, batch_size=1662, shuffle=True)
        return train_dl

    def prepare_test_dataset(path):
        test = CSVDataset(path)
        test_dl = DataLoader(test, batch_size=1024, shuffle=False)
        return test_dl


    train_dl_42 = prepare_train_dataset('/dataset/merged_train_data.csv')
    test_dl_42 = prepare_test_dataset('/dataset/merged_test_data.csv')

    train_dl_28 = prepare_train_dataset('/dataset/merged_train_data_28.csv')
    test_dl_28 = prepare_test_dataset('/dataset/merged_test_data_28.csv')

    train_dl_14 = prepare_train_dataset('/dataset/merged_train_data_14.csv')
    test_dl_14 = prepare_test_dataset('/dataset/merged_test_data_14.csv')

class FraudDetectionMLP(Module):
    def __init__(self, n_inputs):
        super(FraudDetectionMLP, self).__init__()
        # Input layer
        self.hidden1 = Linear(n_inputs, 57)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
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

    # Define loss function and optimizer
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

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


# Run the app
if __name__ == "__main__":
    df = data_ingestion()  # Ensure you have a function or code to load your DataFrame
    main()