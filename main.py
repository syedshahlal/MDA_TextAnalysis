import streamlit as st
from src.data_ingestion import data_ingestion
from src.data_preprocessing import data_preprocessing, split_data, data_resampling, fin_ratio
from src.model import FraudDetectionMLP, run_training_evaluations
from src.dataset import prepare_train_dataset, prepare_test_dataset
from src.utils import plot_misstatements,  check_balance

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
        
        
    if 'cutoff_year' in st.session_state:
        train_df, test_df = split_data(df_filtered, st.session_state['cutoff_year'])
        X_train, y_train = train_df.drop(['misstate'], axis=1), train_df['misstate']
        X_test, y_test = test_df.drop(['misstate'], axis=1), test_df['misstate']
        plot_misstatements(df_filtered, cutoff_year)
        # Store test data in session state here, right after splitting
        st.session_state['X_test'] = X_test
        st.session_state['y_test'] = y_test

        st.header("2. Select the Resampling Strategy")
        resampling_strategy = st.radio(
            "Resampling Strategy",
            ('Random Under Sampling (RUS)', 'Random Over Sampling (ROS)'),
            key="resampling_strategy"
        )

        if st.button('Apply Resampling Strategy'):
            X_train_resampled, y_train_resampled, X_test, y_test = data_resampling(df, cutoff_year, resampling_strategy)
            merged_train_data, merged_test_data, merged_train_data_28, merged_test_data_28, merged_train_data_14, merged_test_data_14, X_train_resampled, y_train_resampled, X_test, y_test = fin_ratio(X_train_resampled, y_train_resampled, X_test, y_test)
            st.session_state['X_train_resampled'] = X_train_resampled
            st.session_state['y_train_resampled'] = y_train_resampled
            st.write(f"{X_train_resampled.shape[0]} samples in the resampled training set.")
            st.write("Class balance:", check_balance(y_train_resampled))
        
        st.header("3. Financial Ratios and Raw Financial Items")
        model_selection = st.radio(
            "Select Model",
            ('All 42 Features', '28 Raw Financial Items', '14 Financial Ratios'),
            key="model_selection"
        )
        
        if 'X_train_resampled' in st.session_state and 'y_train_resampled' in st.session_state:
            X_train_resampled = st.session_state['X_train_resampled']
            y_train_resampled = st.session_state['y_train_resampled']
            X_test = st.session_state['X_test']  
            y_test = st.session_state['y_test']

            # Option to trigger training and evaluation
            if st.button("Train and Evaluate Model"):
                # Determine the number of features based on the selection
                num_features = 42 if model_selection == 'All 42 Features' else 28 if model_selection == '28 Raw Financial Items' else 14

                # Use fin_ratio to prepare datasets according to the selection
                merged_train_data, merged_test_data, merged_train_data_28, merged_test_data_28, merged_train_data_14, merged_test_data_14, _, _, _, _ = fin_ratio(X_train_resampled, y_train_resampled, X_test, y_test)
                
                # Select the correct dataset based on model selection
                train_path = '/dataset/merged_train_data.csv' if model_selection == 'All 42 Features' else '/dataset/merged_train_data_28.csv' if model_selection == '28 Raw Financial Items' else '/dataset/merged_train_data_14.csv'
                test_path = '/dataset/merged_test_data.csv' if model_selection == 'All 42 Features' else '/dataset/merged_test_data_28.csv' if model_selection == '28 Raw Financial Items' else '/dataset/merged_test_data_14.csv'

                # Prepare the DataLoader objects
                train_dl = prepare_train_dataset(train_path)
                test_dl = prepare_test_dataset(test_path)

                
                # Initialize the model with the correct number of input features
                model = FraudDetectionMLP(num_features)

                # Assuming run_training_evaluations is adjusted to take DataLoader objects directly
                average_auc = run_training_evaluations(model, train_dl, test_dl, num_features)
                st.write(f"Average AUC: {average_auc:.4f}")
                st.write(f"Standard Deviation of AUC: {auc_std_dev:.4f}")
 
# Run the app
if __name__ == "__main__":
    # df = data_ingestion()  # Ensure you have a function or code to load your DataFrame
    main()