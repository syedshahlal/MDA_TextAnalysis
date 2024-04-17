import streamlit as st
from src.data_ingestion import data_ingestion
from src.data_preprocessing import data_preprocessing, split_data, data_resampling, fin_ratio
from src.model import FraudDetectionMLP, run_training_evaluations
from src.dataset import prepare_train_dataset, prepare_test_dataset
from src.utils import plot_misstatements,  check_balance, plot_auc
from src.model_utils import save_model, load_model, model_exists


def main():
    st.title("Accounting Fraud Detection")
    st.header('Using Textual and Financial Data from SEC 10-k Filings')
    
    # Data ingestion and preprocessing
    df = data_ingestion()
    df = data_preprocessing(df)
    df_filtered = df[(df['fyear'] >= 2000) & (df['fyear'] <= 2019)]
    
    st.header("1. Select Train and Test Period for Training and Testing Data")
    period_options = {
        "Train period (2000, 2004) and test period (2005, 2019)": ((2000, 2004), (2005, 2019)),
        "Train period (2001, 2005) and test period (2006, 2019)": ((2001, 2005), (2006, 2019)),
        "Train period (2002, 2006) and test period (2007, 2019)": ((2002, 2006), (2007, 2019)),
        "Train period (2003, 2007) and test period (2008, 2019)": ((2003, 2007), (2008, 2019))

    }

    # Use radio buttons for selection
    selected_period_key = st.radio(
        "Choose Training and Testing Periods",
        list(period_options.keys()),
        index=0  # Default to the first option
    )

    # Extract the selected periods
    train_period, test_period = period_options[selected_period_key]

    # Confirm the selected periods and split the data accordingly
    if st.button('Confirm Periods'):
        st.session_state['train_period'] = train_period
        st.session_state['test_period'] = test_period

        st.write(f"Training period selected: {train_period}")
        st.write(f"Testing period selected: {test_period}")
        
        
    if 'train_period' in st.session_state and 'test_period' in st.session_state:
        train_period = st.session_state['train_period']
        test_period = st.session_state['test_period']

        train_df, test_df = split_data(df_filtered, train_period, test_period)
        X_train, y_train = train_df.drop(['misstate'], axis=1), train_df['misstate']
        X_test, y_test = test_df.drop(['misstate'], axis=1), test_df['misstate']
        # Assume plot_misstatements is modified to accept a period instead of a cutoff year
        plot_misstatements(df_filtered, train_period, test_period)
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
            X_train_resampled, y_train_resampled, X_test, y_test = data_resampling(
                df, train_period, test_period, resampling_strategy
            )
            # Assume fin_ratio is modified to accept resampled data directly
            merged_train_data, merged_test_data, merged_train_data_28, merged_test_data_28, merged_train_data_14, merged_test_data_14, X_train_resampled, y_train_resampled, X_test, y_test = fin_ratio(
                X_train_resampled, y_train_resampled, X_test, y_test
            )
            st.session_state['X_train_resampled'] = X_train_resampled
            st.session_state['y_train_resampled'] = y_train_resampled
            st.write(f"{X_train_resampled.shape[0]} samples in the resampled training set.")
            st.write("Class balance:", check_balance(y_train_resampled))
        
        st.header("3. Financial Ratios and Raw Financial Items along with Textual Features")
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
            if st.button("Run model"):
                # Determine the number of features based on the selection
                num_features = 42 if model_selection == 'All 42 Features' else 28 if model_selection == '28 Raw Financial Items' else 14
                train_path = '/dataset/merged_train_data.csv' if model_selection == 'All 42 Features' else '/dataset/merged_train_data_28.csv' if model_selection == '28 Raw Financial Items' else '/dataset/merged_train_data_14.csv'
                test_path = '/dataset/merged_test_data.csv' if model_selection == 'All 42 Features' else '/dataset/merged_test_data_28.csv' if model_selection == '28 Raw Financial Items' else '/dataset/merged_test_data_14.csv'

                # Prepare the DataLoader objects
                train_dl = prepare_train_dataset(train_path)
                test_dl = prepare_test_dataset(test_path)

                model_dir = r'D:\University\UB\Research_SEC\MDA_TextAnalysis\model'  # Replace with the actual directory path
                if model_exists(num_features, model_dir):
                    # Model exists, load the model and its metadata
                    model, metadata = load_model(num_features, model_dir)
                    st.write("Loaded pre-trained model.")
                    
                    # Accessing the metadata
                    auc_values = metadata.get('auc_values', [])
                    average_auc = metadata.get('average_auc')
                    std_auc = metadata.get('std_auc')
                    
                    # Displaying the metadata
                    st.write(f"Average AUC: {average_auc:.4f}")
                    if std_auc is not None:
                        st.write(f"Standard Deviation of AUC: {std_auc:.4f}")
                    else:
                        pass
                    plot_auc(auc_values)

                else:
                    # Model doesn't exist, proceed to train and save
                    model = FraudDetectionMLP(num_features)

                    st.write(f"Trainset: {train_dl.dataset.X.shape} samples, Testset: {test_dl.dataset.X.shape} samples")
                

                    # Assuming run_training_evaluations is adjusted to take DataLoader objects directly
                    metadata = run_training_evaluations(model, train_dl, test_dl)
                    auc_values = metadata.get('auc_values', [])
                    average_auc = metadata.get('average_auc', 0)
                    auc_std_dev = metadata.get('auc_std_dev', 0)
                    save_model(model, num_features, metadata, model_dir="model")
                    st.write(f"Model trained. Average AUC: {average_auc:.4f}, Standard Deviation of AUC: {auc_std_dev:.4f}")
                    plot_auc(auc_values)
 
# Run the app
if __name__ == "__main__":
    # df = data_ingestion()  # Ensure you have a function or code to load your DataFrame
    main()