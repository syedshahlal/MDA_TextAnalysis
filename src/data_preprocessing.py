import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import os
import streamlit as st

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

    directory = r"D:\University\UB\Research_SEC\MDA_TextAnalysis\dataset"

    # Save CSV files
    merged_train_data.to_csv(os.path.join(directory, 'merged_train_data.csv'), index=False)
    merged_test_data.to_csv(os.path.join(directory, 'merged_test_data.csv'), index=False)
    merged_train_data_28.to_csv(os.path.join(directory, 'merged_train_data_28.csv'), index=False)
    merged_test_data_28.to_csv(os.path.join(directory, 'merged_test_data_28.csv'), index=False)
    merged_train_data_14.to_csv(os.path.join(directory, 'merged_train_data_14.csv'), index=False)
    merged_test_data_14.to_csv(os.path.join(directory, 'merged_test_data_14.csv'), index=False)

    return (merged_train_data, merged_test_data, merged_train_data_28, merged_test_data_28, 
            merged_train_data_14, merged_test_data_14, X_train_resampled, y_train_resampled, X_test, y_test)