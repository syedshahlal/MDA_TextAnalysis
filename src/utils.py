import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

@st.cache_data
def plot_misstatements(df, train_period, test_period):
    # Ensure df is filtered for years within the given period
    filtered_df = df[(df['fyear'] >= train_period[0]) & (df['fyear'] <= test_period[1])]
    
    # Dynamically split based on the train_period and test_period
    training_df = filtered_df[(filtered_df['fyear'] >= train_period[0]) & (filtered_df['fyear'] <= train_period[1])]
    testing_df = filtered_df[(filtered_df['fyear'] >= test_period[0]) & (filtered_df['fyear'] <= test_period[1])]

    # Group by fiscal year and sum misstatements
    misstatements_by_fyear = filtered_df.groupby('fyear')['misstate'].sum()
    misstatements_by_fyear_training = training_df.groupby('fyear')['misstate'].sum()
    misstatements_by_fyear_testing = testing_df.groupby('fyear')['misstate'].sum()

    # Calculate total misstatements
    total_misstatements = misstatements_by_fyear.sum()
    total_misstatements_training = misstatements_by_fyear_training.sum()
    total_misstatements_testing = misstatements_by_fyear_testing.sum()

    colors = ['rgb(0, 88, 186)',  # Darker Blue
              'rgb(229, 106, 81)']  # Red
    
    # Pie Chart
    labels = ['Training Data Misstatements', 'Testing Data Misstatements']
    sizes = [total_misstatements_training, total_misstatements_testing]
    
    fig1 = go.Figure(data=[go.Pie(labels=labels, values=sizes, pull=[0.1, 0], marker_colors=colors)])
    fig1.update_traces(textinfo='value+percent', textfont_size=14, insidetextfont={'color': 'white', 'family': 'Arial Black', 'size': 14})
    fig1.update_layout(title_text='Proportion of Misstatements: Training vs. Testing')
    
    st.plotly_chart(fig1)

    # Bar Graph
    fig2 = px.bar(x=misstatements_by_fyear.index, y=misstatements_by_fyear.values, labels={'x': 'Fiscal Year', 'y': 'Number of Misstatements'})
    # Use train_period and test_period to add annotations for the training and testing periods
    fig2.add_vrect(x0=train_period[0], x1=train_period[1], fillcolor="blue", opacity=0.1, line_width=0, annotation_text="Training Period", annotation_position="top left")
    fig2.add_vrect(x0=test_period[0], x1=test_period[1], fillcolor="red", opacity=0.1, line_width=0, annotation_text="Testing Period", annotation_position="top right")
    fig2.update_layout(title_text='Misstatements by Fiscal Year', xaxis_title='Fiscal Year', yaxis_title='Number of Misstatements')
    fig2.update_xaxes(tickmode='linear', dtick=1)

    st.plotly_chart(fig2)

    return total_misstatements, total_misstatements_training, total_misstatements_testing

def check_balance(y_train_resampled):
    """Check if the dataset is balanced."""
    class_counts = pd.Series(y_train_resampled).value_counts()
    is_balanced = class_counts.min() / class_counts.max() >= 0.8
    return is_balanced


def plot_auc(auc_values):
    """Plot the AUC values using Plotly."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1, len(auc_values) + 1)), y=auc_values,
                             mode='lines+markers', 
                             marker=dict(color='red', size=10),
                             line=dict(color='red')))
    
    fig.update_layout(title='AUC Values Over Trials',
                      xaxis_title='Number of Trials',
                      yaxis_title='AUC',
                      template='plotly_white')
    
    st.plotly_chart(fig)
