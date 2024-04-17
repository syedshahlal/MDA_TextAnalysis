import pandas as pd
import streamlit as st

@st.cache_data()
def data_ingestion():
    # Load the dataset
    url = "https://media.githubusercontent.com/media/syedshahlal/MDA_TextAnalysis/main/dataset/merged_compustat_and_labels.csv"
    df = pd.read_csv(url)
    return df