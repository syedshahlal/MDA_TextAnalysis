import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from torch.nn import Linear, ReLU, Tanh, Sigmoid, Module, BCELoss
from torch.optim import SGD, Adam, lr_scheduler
import torch.nn as nn
from torch.nn.init import kaiming_uniform_, xavier_uniform_
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit, Probit
import time
import copy
import streamlit as st
import matplotlib.pyplot as plt

# Set the random seed for reproducibility
torch.manual_seed(0)
def main():
    st.title("Fraud Detection in Financial Statements")

    # Data ingestion and preprocessing
    df = data_ingestion()
    df = data_preprocessing(df)
    df_filtered = df[(df['fyear'] >= 1990) & (df['fyear'] <= 2019)]
    
    st.header("1. Select Cutoff Year for Training and Testing Data")
    cutoff_year = st.slider("Cutoff Year", min_value=1990, max_value=2019, value=st.session_state.get('cutoff_year', 2002), step=1)

    if st.button('Confirm Cutoff Year'):
        st.session_state['cutoff_year'] = cutoff_year
        plot_misstatements(df_filtered, cutoff_year)
        

    if 'cutoff_year' in st.session_state:
        train_df, test_df = split_data(df_filtered, st.session_state['cutoff_year'])
        st.write(f"Training data shape: {train_df.shape}")
        st.write(f"Testing data shape: {test_df.shape}")
        X_train, y_train = train_df.drop(['misstate'], axis=1), train_df['misstate']
        X_test, y_test = test_df.drop(['misstate'], axis=1), test_df['misstate']

        st.header("2. Select the Resampling Strategy")
        resampling_strategy = st.radio(
            "Resampling Strategy",
            ('Random Under Sampling (RUS)', 'Random Over Sampling (ROS)'),
            key="resampling_strategy"
        )

        if st.button('Apply Resampling Strategy'):
            # Make sure to correctly implement data_resampling to handle resampling based on the selected strategy
            X_train_resampled, y_train_resampled, X_test, y_test = data_resampling(df, cutoff_year, resampling_strategy)
            st.session_state['X_train_resampled'] = X_train_resampled
            st.session_state['y_train_resampled'] = y_train_resampled

        if 'X_train_resampled' in st.session_state and 'y_train_resampled' in st.session_state:
            X_train_resampled = st.session_state['X_train_resampled']
            y_train_resampled = st.session_state['y_train_resampled']
            st.write(f"{X_train_resampled.shape[0]} samples in the resampled training set.")
            st.write("Class balance:", check_balance(y_train_resampled))

            st.header("3. Financial Ratios and Raw Financial Items")
            model_selection = st.radio(
                "Select Model",
                ('All 42 Features', '28 Raw Financial Items', '14 Financial Ratios'),
                key="model_selection"
            )
        
@st.cache_data
def data_ingestion():
    # Load the dataset
    url = "https://media.githubusercontent.com/media/syedshahlal/MDA_TextAnalysis/main/dataset/merged_compustat_and_labels.csv"
    df = pd.read_csv(url)
    return df

@st.cache_data
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

@st.cache_data
def split_data(df_filtered, cutoff_year):
    """
    Split the dataset into training and testing sets based on a cutoff year.
    
    Parameters:
    - df_filtered: DataFrame containing the filtered data.
    - cutoff_year: Integer representing the year to split the data on.
    
    Returns:
    - train_df: DataFrame containing the training data.
    - test_df: DataFrame containing the testing data.
    """
    # Split the DataFrame into training and testing based on the cutoff year
    train_df = df_filtered[df_filtered['fyear'] <= cutoff_year]
    test_df = df_filtered[df_filtered['fyear'] > cutoff_year]
    
    return train_df, test_df


@st.cache_data
def plot_misstatements(df, cutoff_year):
    # Ensure df is filtered for years from 1990 to 2019
    filtered_df = df[(df['fyear'] >= 1990) & (df['fyear'] <= 2019)]
    
    # Dynamically split based on the cutoff year
    training_df = filtered_df[filtered_df['fyear'] <= cutoff_year]
    testing_df = filtered_df[filtered_df['fyear'] > cutoff_year]

    # Group by fiscal year and sum misstatements
    misstatements_by_fyear = filtered_df.groupby('fyear')['misstate'].sum()
    misstatements_by_fyear_training = training_df.groupby('fyear')['misstate'].sum()
    misstatements_by_fyear_testing = testing_df.groupby('fyear')['misstate'].sum()

    # Calculate total misstatements
    total_misstatements = misstatements_by_fyear.sum()
    total_misstatements_training = misstatements_by_fyear_training.sum()
    total_misstatements_testing = misstatements_by_fyear_testing.sum()
    
    # Create pie chart
    fig1, ax1 = plt.subplots()
    labels = ['Training Data Misstatements', 'Testing Data Misstatements']
    sizes = [total_misstatements_training, total_misstatements_testing]
    colors = ['#ff9999', '#66b3ff']
    explode = (0.1, 0)  # explode the first slice

    # Custom autopct function to show both value and percentage
    def autopct_format(values):
        def my_format(pct):
            total = sum(values)
            val = int(round(pct*total/100.0))
            return '{v:d} ({p:.2f}%)'.format(v=val, p=pct)
        return my_format

    ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct=autopct_format(sizes), shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title('Proportion of Misstatements: Training vs. Testing')
    st.pyplot(fig1) # Show pie chart

    # Clear figure for next plot
    plt.clf()

    # Create bar graph for total misstatements by fiscal year
    fig2, ax2 = plt.subplots()
    misstatements_by_fyear = filtered_df.groupby('fyear')['misstate'].sum()
    ax2.bar(misstatements_by_fyear.index, misstatements_by_fyear.values, color='skyblue')
    ax2.axvline(x=cutoff_year, color='red', linestyle='--', label='Cutoff Year')  # Adjusting cutoff_year visualization
    ax2.set_xlabel('Fiscal Year')
    ax2.set_ylabel('Number of Misstatements')
    ax2.set_title('Misstatements by Fiscal Year')

    # Set the tick labels on the x-axis at a 45-degree angle
    plt.xticks(rotation=45)

    ax2.legend()
    st.pyplot(fig2)

    return total_misstatements, total_misstatements_training, total_misstatements_testing

@st.cache_data
def data_resampling(df, cutoff_year, resampling_strategy):
    """
    Splits the data based on a cutoff year and applies the selected resampling strategy to balance the training set.
    """
    features = ['act', 'ap', 'at', 'ceq', 'che', 'cogs', 'csho', 'dlc', 'dltis', 'dltt', 'dp', 'ib',
                'invt', 'ivao', 'ivst', 'lct', 'lt', 'ni', 'ppegt', 'pstk', 're', 'rect', 'sale', 'sstk',
                'txp', 'txt', 'xint', 'prcc_f', 'dch_wc', 'ch_rsst', 'dch_rec', 'dch_inv', 'soft_assets',
                'ch_cs', 'ch_cm', 'ch_roa', 'bm', 'dpi', 'reoa', 'EBIT', 'ch_fcf', 'issue']
    
    train_data = df[df['fyear'] <= cutoff_year]
    test_data = df[df['fyear'] > cutoff_year]

    X_train = train_data[features]
    y_train = train_data['misstate']
    X_test = test_data[features]
    y_test = test_data['misstate']

    if resampling_strategy == 'Random Under Sampling (RUS)':
        sampler = RandomUnderSampler()
    elif resampling_strategy == 'Random Over Sampling (ROS)':
        sampler = RandomOverSampler()

    X_train_resampled, y_train_resampled = sampler.fit_resample(X_train, y_train)

    return X_train_resampled, y_train_resampled, X_test, y_test

def check_balance(y_train_resampled):
    """
    Check if the dataset is balanced based on the resampled training labels.
    Returns True if balanced, False otherwise.
    """
    # Count the occurrences of each class in the resampled training dataset
    class_counts = pd.Series(y_train_resampled).value_counts()
    
    # Determine if the dataset is balanced (assuming a threshold for "balanced" at 80/20 distribution)
    is_balanced = class_counts.min() / class_counts.max() >= 0.8
    
    return is_balanced

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

    return merged_train_data, merged_test_data, merged_train_data_28, merged_test_data_28, merged_train_data_14, merged_test_data_14, X_train_resampled, y_train_resampled, X_test, y_test

# class CSVDataset(Dataset):
#     #Constructor for initially loading
#     def __init__(self, path):
#         df = read_csv(path, header=0)
#         self.X = df.values[0:, :-1]
#         self.y = df.values[0:, -1]
#         self.X = self.X.astype('float32')
#         self.y = self.y.astype('float32')
#         self.y = self.y.reshape((len(self.y), 1))

#         print(self.X.shape)
#         print(self.y.shape)

#     # Get the number of rows in the dataset
#     def __len__(self):
#         return len(self.X)
#     # Get a row at an index
#     def __getitem__(self,idx):
#         return [self.X[idx], self.y[idx]]

#     def prepare_train_dataset(path):
#         train = CSVDataset(path)
#         train_dl = DataLoader(train, batch_size=1662, shuffle=True)
#         return train_dl

#     def prepare_test_dataset(path):
#         test = CSVDataset(path)
#         test_dl = DataLoader(test, batch_size=1024, shuffle=False)
#         return test_dl


#     train_dl_42 = prepare_train_dataset('/dataset/merged_train_data.csv')
#     test_dl_42 = prepare_test_dataset('/dataset/merged_test_data.csv')

#     train_dl_28 = prepare_train_dataset('/dataset/merged_train_data_28.csv')
#     test_dl_28 = prepare_test_dataset('/dataset/merged_test_data_28.csv')

#     train_dl_14 = prepare_train_dataset('/dataset/merged_train_data_14.csv')
#     test_dl_14 = prepare_test_dataset('/dataset/merged_test_data_14.csv')

# class FraudDetectionMLP(Module):
#     def __init__(self, n_inputs):
#         super(FraudDetectionMLP, self).__init__()
#         # Input layer
#         self.hidden1 = Linear(n_inputs, 57)
#         kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
#         self.act1 = ReLU()
#         # Second (hidden) layer
#         self.hidden2 = Linear(57, 22)
#         kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
#         self.act2 = ReLU()
#         # Third (hidden) layer
#         self.hidden3 = Linear(22, 107)
#         kaiming_uniform_(self.hidden3.weight, nonlinearity='relu')
#         self.act3 = ReLU()
#         # Fourth (hidden) layer
#         self.hidden4 = Linear(107, 202)
#         kaiming_uniform_(self.hidden4.weight, nonlinearity='relu')
#         self.act4 = Sigmoid()
#         # Fifth (hidden) layer
#         self.hidden5 = Linear(202, 162)
#         kaiming_uniform_(self.hidden5.weight, nonlinearity='relu')
#         self.act5 = Sigmoid()
#         # Output layer
#         self.hidden6 = Linear(162,1)
#         xavier_uniform_(self.hidden6.weight)
#         self.act6 = Sigmoid()

#     def forward(self, X):
#         # Input to the first hidden layer
#         X = self.hidden1(X)
#         X = self.act1(X)
#         # Second hidden layer
#         X = self.hidden2(X)
#         X = self.act2(X)
#         # Third hidden layer
#         X = self.hidden3(X)
#         X = self.act3(X)
#         # Fourth hidden layer
#         X = self.hidden4(X)
#         X = self.act4(X)
#         # Fifth hidden layer
#         X = self.hidden5(X)
#         X = self.act5(X)
#         # Output layer
#         X = self.hidden6(X)
#         X = self.act6(X)
#         return X
    
# def train_model(model, train_dl, num_epochs):

#     # Define loss function and optimizer
#     criterion = torch.nn.BCELoss()
#     optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         for inputs, labels in train_dl:
#             optimizer.zero_grad()
#             outputs = model(inputs.float())
#             labels = labels.float()
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item() * inputs.size(0)
#         epoch_loss = running_loss / len(train_dl.dataset)

# def evaluate_model(model, test_dl):
#     model.eval()
#     y_true = []
#     y_pred = []
#     with torch.no_grad():
#         for inputs, labels in test_dl:
#             outputs = model(inputs.float())
#             y_true.extend(labels.numpy())
#             y_pred.extend(outputs.numpy().flatten())
#     auc = roc_auc_score(y_true, y_pred)
#     return auc

# model = FraudDetectionMLP(42)
# train_model(model, train_dl_42, num_epochs=150)
# evaluate_model(model, test_dl_42)

# auc_values = []

# # Perform 10 training runs
# for i in range(10):
#     model_42 = FraudDetectionMLP(42)
#     train_model(model_42, train_dl_42, num_epochs=150)

#     # Evaluate the model
#     model_42.eval()
#     y_true = []
#     y_pred = []
#     with torch.no_grad():
#         for inputs, labels in test_dl_42:
#             outputs = model_42(inputs.float())
#             y_true.extend(labels.numpy())
#             y_pred.extend(outputs.numpy().flatten())
#     auc = roc_auc_score(y_true, y_pred)
#     auc_values.append(auc)  # Append AUC to the list
#     print(f"Run {i+1}: AUC = {auc:.4f}")

# # Calculate the average AUC
# average_auc = np.mean(auc_values)
# print(f"\nAverage AUC: {average_auc:.4f}")

# # Calculate the std AUC
# auc_std_dev = np.std(auc_values)
# print(f"Standard Deviation of AUC: {auc_std_dev:.4f}")

# # 28 raw financial items model
# model_28 = FraudDetectionMLP(28)

# train_model(model_28, train_dl_28, num_epochs=150)
# evaluate_model(model_28, test_dl_28)

# auc_values_28 = []

# # Perform 10 training runs
# for i in range(10):
#     model_28 = FraudDetectionMLP(28)
#     train_model(model_28, train_dl_28, num_epochs=150)

#     # Evaluate the model
#     model_28.eval()
#     y_true = []
#     y_pred = []
#     with torch.no_grad():
#         for inputs, labels in test_dl_28:
#             outputs = model_28(inputs.float())
#             y_true.extend(labels.numpy())
#             y_pred.extend(outputs.numpy().flatten())
#     auc_28 = roc_auc_score(y_true, y_pred)
#     auc_values_28.append(auc_28)  # Append AUC to the list
#     print(f"Run {i+1}: AUC = {auc_28:.4f}")

# # Calculate the average AUC
# average_auc_28 = np.mean(auc_values_28)
# print(f"\nAverage AUC: {average_auc_28:.4f}")

# # Calculate the std AUC
# auc_std_dev_28 = np.std(auc_values_28)
# print(f"Standard Deviation of AUC: {auc_std_dev_28:.4f}")

# # 14 financial ratios model
# model_14 = FraudDetectionMLP(14)

# train_model(model_14, train_dl_14, num_epochs=150)
# evaluate_model(model_14, test_dl_14)

# auc_values_14 = []

# # Perform 10 training runs
# for i in range(10):
#     model_14 = FraudDetectionMLP(14)
#     train_model(model_14, train_dl_14, num_epochs=150)

#     # Evaluate the model
#     model_14.eval()
#     y_true = []
#     y_pred = []
#     with torch.no_grad():
#         for inputs, labels in test_dl_14:
#             outputs = model_14(inputs.float())
#             y_true.extend(labels.numpy())
#             y_pred.extend(outputs.numpy().flatten())
#     auc_14 = roc_auc_score(y_true, y_pred)
#     auc_values_14.append(auc_14)  # Append AUC to the list
#     print(f"Run {i+1}: AUC = {auc_14:.4f}")

# # Calculate the average AUC
# average_auc_14 = np.mean(auc_values_14)
# print(f"\nAverage AUC: {average_auc_14:.4f}")

# auc_std_dev_14 = np.std(auc_values_14)
# print(f"Standard Deviation of AUC: {auc_std_dev_14:.4f}")


# Run the app
if __name__ == "__main__":
    # df = data_ingestion()  # Ensure you have a function or code to load your DataFrame
    main()