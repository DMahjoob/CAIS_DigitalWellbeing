# pip3 install autogluon.tabular
pip install torch lightgbm fastai
from autogluon.tabular import TabularDataset, TabularPredictor

import pandas as pd
import numpy as np
from datetime import datetime

# Load your data
sensing_data = pd.read_csv('/Users/rachitajain/Desktop/cais24/Sensing/sensing.csv')
demographics_data = pd.read_csv('/Users/rachitajain/Desktop/cais24/Demographics/demographics.csv')

# Merge sensing and demographics data
df = sensing_data.merge(
    demographics_data,
    on='uid',  # common column between both datasets
    how='inner'  # only keep rows that exist in both datasets
)

# Now you can check the shape of your merged dataset
print("Original sensing data shape:", sensing_data.shape)
print("Demographics data shape:", demographics_data.shape)
print("Merged data shape:", df.shape)

print("\nMerged columns:", df.columns.tolist())


# Create quarter-based split function
def create_quarter_based_split(df, test_size=0.2, random_state=42):
    # Convert YYYYMMDD to datetime
    df['date'] = pd.to_datetime(df['day'], format='%Y%m%d')
    
    # Extract quarter information
    df['quarter'] = df['date'].dt.quarter
    
    # Get unique quarters in the dataset
    unique_quarters = df['quarter'].unique()
    
    # Randomly select quarters for test set
    np.random.seed(random_state)
    num_test_quarters = max(1, int(len(unique_quarters) * test_size))
    test_quarters = np.random.choice(unique_quarters, size=num_test_quarters, replace=False)
    
    # Split data into train and test
    test_data = df[df['quarter'].isin(test_quarters)].copy()
    train_data = df[~df['quarter'].isin(test_quarters)].copy()
    
    # Drop temporary columns
    train_data = train_data.drop(['date', 'quarter'], axis=1)
    test_data = test_data.drop(['date', 'quarter'], axis=1)
    
    return train_data, test_data

# Convert to TabularDataset and split
df = TabularDataset(df)
train_data, test_data = create_quarter_based_split(df)

# You can subsample if needed
subsample_size = 500
train_data = train_data.sample(n=subsample_size, random_state=0)

# Define your label column
label = 'sleep_duration'  # Replace with your actual target column name

# Train the model
predictor = TabularPredictor(
    label=label,
    eval_metric='rmse'  # Root Mean Square Error for regression
).fit(
    train_data,
    time_limit=600,  # 10 minutes
    presets='best_quality'
)


