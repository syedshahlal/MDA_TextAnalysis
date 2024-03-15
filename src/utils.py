import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

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

def check_balance(y_train_resampled):
    """Check if the dataset is balanced."""
    class_counts = pd.Series(y_train_resampled).value_counts()
    is_balanced = class_counts.min() / class_counts.max() >= 0.8
    return is_balanced



