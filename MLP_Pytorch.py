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