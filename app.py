import numpy as np
import pandas as pd
import torch
from numpy import vstack
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from imblearn.under_sampling import RandomUnderSampler
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
    
    # Data ingestion
    df = data_ingestion()  # Load your DataFrame
    
    # Ensure df is available by checking its shape or similar
    st.write(f"Data Loaded: {df.shape[0]} rows and {df.shape[1]} columns.")
    
    # Data preprocessing
    df = data_preprocessing(df)
    
    # Define the minimum and maximum years for the slider, based on the preprocessed df
    min_year, max_year = int(df['fyear'].min()), int(df['fyear'].max())
    
    # Sliders for selecting the training and testing period
    train_start, train_end = st.slider("Select Training Period", min_value=min_year, max_value=max_year, value=(min_year, 2002), step=1)
    test_start, test_end = st.slider("Select Testing Period", min_value=min_year, max_value=max_year, value=(2003, max_year), step=1)
    
    # Check for overlap between train and test periods
    if train_end >= test_start:
        st.error("Training period must end before the testing period starts.")
    else:
        if st.button('Split Data'):
            # Splitting the data according to the selected periods
            X_train_resampled, y_train_resampled, X_test, y_test = data_splitting(df, (train_start, train_end), (test_start, test_end))
            st.success("Data Splitting Complete.")
            
            # Optionally display shapes of the datasets or other information
            st.write(f"Training Data Shape: {X_train_resampled.shape}")
            st.write(f"Testing Data Shape: {X_test.shape}")
            
            # Optional: data plotting to visualize the distribution or summary
            total_misstatements, total_misstatements_training, total_misstatements_testing = data_plottings(df)
            st.write(f"Total Misstatements: {total_misstatements}")
            st.write(f"Training Period Misstatements: {total_misstatements_training}")
            st.write(f"Testing Period Misstatements: {total_misstatements_testing}")




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

def data_plottings(df):
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
    plt.figure(figsize=(8, 4))
    misstatements_by_fyear.plot(kind='bar')
    plt.xlabel('Fiscal Year')
    plt.ylabel('Number of Misstatements')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

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
    train_data = df[(df['fyear'] >= train_period[0]) & (df['fyear'] <= train_period[1])]
    test_data = df[(df['fyear'] >= test_period[0]) & (df['fyear'] <= test_period[1])]

    # Extract features (X) and target variable (y) for training and testing
    X_train = train_data[raw_items_28_financial_ratios_14]
    y_train = train_data['misstate']

    X_test = test_data[raw_items_28_financial_ratios_14]
    y_test = test_data['misstate']

    # Undersampling the data
    rus = RandomUnderSampler()
    X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)

    return X_train_resampled, y_train_resampled, X_test, y_test


# Run the app
if __name__ == "__main__":
    main()