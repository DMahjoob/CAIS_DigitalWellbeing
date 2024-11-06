# pip install autogluon.tabular[all]
from autogluon.tabular import TabularDataset, TabularPredictor

import pandas as pd
import numpy as np
from datetime import datetime

# Load your data
df = pd.read_csv('/Users/rachitajain/Desktop/cais24/Sensing/sensing.csv')

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
label = 'your_target_column'  # Replace with your actual target column name

# Train the model
predictor = TabularPredictor(label=label).fit(train_data)

# Make predictions
predictions = predictor.predict(test_data)

# Evaluate performance
perf = predictor.evaluate(test_data)
print(f"Model performance: {perf}")

label = 'class'
print(f"Unique classes: {list(train_data[label].unique())}")

predictor = TabularPredictor(label=label).fit(train_data)

y_pred = predictor.predict(test_data)
y_pred.head()  # Predictions

y_pred_proba = predictor.predict_proba(test_data)
y_pred_proba.head()  # Prediction Probabilities

#Evaluation
predictor.evaluate(test_data)
predictor.leaderboard(test_data)

predictor.path  # The path on disk where the predictor is saved

# Load the predictor by specifying the path it is saved to on disk.
# You can control where it is saved to by setting the `path` parameter during init
predictor = TabularPredictor.load(predictor.path)

print("AutoGluon infers problem type is: ", predictor.problem_type)
print("AutoGluon identified the following types of features:")
print(predictor.feature_metadata)

test_data_transform = predictor.transform_features(test_data)
test_data_transform.head()

predictor.feature_importance(test_data)

predictor.model_best

predictor.model_names()

time_limit = 60  # for quick demonstration only, you should set this to longest time you are willing to wait (in seconds)
metric = 'roc_auc'  # specify your evaluation metric here
predictor = TabularPredictor(label, eval_metric=metric).fit(train_data, time_limit=time_limit, presets='best_quality')

predictor.leaderboard(test_data)

age_column = 'age'
train_data[age_column].head()

predictor_age = TabularPredictor(label=age_column, path="agModels-predictAge").fit(train_data, time_limit=60)

predictor_age.evaluate(test_data)

predictor_age.leaderboard(test_data)