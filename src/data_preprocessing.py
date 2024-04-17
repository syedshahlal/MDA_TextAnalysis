import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
import os
import streamlit as st
import torch
from src.model import FraudDetectionMLP


def model_exists(num_features, model_dir="model"):
    model_path = f"{model_dir}/fraud_detection_model_{num_features}_features.pth"
    return os.path.isfile(model_path)

def save_model(model, num_features, model_dir="model"):
    model_path = f"{model_dir}/fraud_detection_model_{num_features}_features.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model with {num_features} features saved to {model_path}")

def load_model(num_features, model_dir="model"):
    model_path = f"{model_dir}/fraud_detection_model_{num_features}_features.pth"
    model = FraudDetectionMLP(num_features)  # You need to initialize the model structure
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


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
def split_data(df, train_period, test_period):
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
    train_df = df[(df['fyear'] >= train_period[0]) & (df['fyear'] <= train_period[1])]
    test_df = df[(df['fyear'] >= test_period[0]) & (df['fyear'] <= test_period[1])]
    return train_df, test_df


@st.cache_data
def data_resampling(df, train_period, test_period, resampling_strategy):
    """
    Splits the data based on selected training and testing periods and applies the selected 
    resampling strategy to balance the training set.

    Parameters:
    - df: DataFrame containing the data.
    - train_period: Tuple representing the start and end year of the training period.
    - test_period: Tuple representing the start and end year of the testing period.
    - resampling_strategy: String representing the resampling strategy ('RUS' or 'ROS').

    Returns:
    - X_train_resampled: Features of the resampled training set.
    - y_train_resampled: Labels of the resampled training set.
    - X_test: Features of the testing set.
    - y_test: Labels of the testing set.
    """
    features = ['act', 'ap', 'at', 'ceq', 'che', 'cogs', 'csho', 'dlc', 'dltis', 'dltt', 'dp', 'ib',
                'invt', 'ivao', 'ivst', 'lct', 'lt', 'ni', 'ppegt', 'pstk', 're', 'rect', 'sale', 'sstk',
                'txp', 'txt', 'xint', 'prcc_f', 'dch_wc', 'ch_rsst', 'dch_rec', 'dch_inv', 'soft_assets',
                'ch_cs', 'ch_cm', 'ch_roa', 'bm', 'dpi', 'reoa', 'EBIT', 'ch_fcf', 'issue']

    # Update to use the train_period and test_period tuples
    train_data = df[(df['fyear'] >= train_period[0]) & (df['fyear'] <= train_period[1])]
    test_data = df[(df['fyear'] >= test_period[0]) & (df['fyear'] <= test_period[1])]

    X_train = train_data[features]
    y_train = train_data['misstate']
    X_test = test_data[features]
    y_test = test_data['misstate']

    # Resampling strategy selection
    if resampling_strategy == 'Random Under Sampling (RUS)':
        sampler = RandomUnderSampler()
    elif resampling_strategy == 'Random Over Sampling (ROS)':
        sampler = RandomOverSampler()

    # Apply resampling
    X_train_resampled, y_train_resampled = sampler.fit_resample(X_train, y_train)

    counter_before = Counter(y_train)
    counter_after = Counter(y_train_resampled)

    # Convert to lists for Plotly
    categories = list(counter_before.keys())
    values_before = list(counter_before.values())
    values_after = list(counter_after.values())

    # Creating a figure with subplots
    fig = go.Figure(data=[
        go.Bar(name='Before Resampling', x=categories, y=values_before, text=values_before,
            marker_color='red', textposition='auto'),
        go.Bar(name='After Resampling', x=categories, y=values_after, text=values_after,
            marker_color='blue', textposition='auto')
    ])

    # Update the layout
    fig.update_layout(
        barmode='group',
        title='Class Distribution Before and After Resampling',
        xaxis_title='Class',
        yaxis_title='Frequency',
        width=800,
        height=450,
    )

    # Set x-axis category order if needed, assuming classes are 0 and 1
    fig.update_xaxes(categoryorder='array', categoryarray=[0, 1])

    # Show the figure
    st.plotly_chart(fig)

    return X_train_resampled, y_train_resampled, X_test, y_test


@st.cache_data
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

    merged_train_data = pd.concat([X_train_resampled, y_train_resampled], axis=1)
    merged_test_data = pd.concat([X_test, y_test], axis=1)

    merged_train_data_28 = pd.concat([X_train_resampled_28, y_train_resampled_28], axis=1)
    merged_test_data_28 = pd.concat([X_test_28, y_test_28], axis=1)

    merged_train_data_14 = pd.concat([X_train_resampled_14, y_train_resampled_14], axis=1)
    merged_test_data_14 = pd.concat([X_test_14, y_test_14], axis=1)

    directory = ".././dataset"

    if not os.path.exists(directory):
     os.makedirs(directory)

    # Save CSV files
    merged_train_data.to_csv(os.path.join(directory, 'merged_train_data.csv'), index=False)
    merged_test_data.to_csv(os.path.join(directory, 'merged_test_data.csv'), index=False)
    merged_train_data_28.to_csv(os.path.join(directory, 'merged_train_data_28.csv'), index=False)
    merged_test_data_28.to_csv(os.path.join(directory, 'merged_test_data_28.csv'), index=False)
    merged_train_data_14.to_csv(os.path.join(directory, 'merged_train_data_14.csv'), index=False)
    merged_test_data_14.to_csv(os.path.join(directory, 'merged_test_data_14.csv'), index=False)

    return (merged_train_data, merged_test_data, merged_train_data_28, merged_test_data_28, 
            merged_train_data_14, merged_test_data_14, X_train_resampled, y_train_resampled, X_test, y_test)