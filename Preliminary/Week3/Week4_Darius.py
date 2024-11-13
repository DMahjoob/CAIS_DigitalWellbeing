import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Print working directory
import os
print(os.getcwd())

# Load Data
demographics_data = pd.read_csv("demographics.csv")
sensing_data = pd.read_csv("sensing_data.csv")
ema_data = pd.read_csv("general_ema.csv")

# Define feature dictionary
# talk about this in paper // how to find the right features
# data analysis correlation/testing pick out
# feed all features into a machine learning model and the model tells us what is relevant
# check feature_importance property (only put ones with positive scores)
feature_dict = {
    'demographics': ['gender', 'race'],
    'activity': {
        'all': ['act_in_vehicle_ep_0', 'act_on_bike_ep_0', 'act_still_ep_0'],
        'android': ['act_on_foot_ep_0', 'act_tilting_ep_0'],
        'ios': ['act_running_ep_0', 'act_walking_ep_0']
    },
    # Add other categories as needed
}
# some features only available on certain OSs
# manually create "ios" / "android" column

# Define function to assign academic quarter
def get_academic_quarter(day):
    if 1 <= day <= 90:
        return 'Q1'
    elif 91 <= day <= 180:
        return 'Q2'
    elif 181 <= day <= 270:
        return 'Q3'
    else:
        return 'Q4'

# Merge datasets
def prepare_merged_dataset(sensing_data, demographics_data, ema_data):
    merged = sensing_data.merge(demographics_data, on='uid', how='inner')
    merged = merged.merge(ema_data, on=['uid', 'day'], how='inner')
    merged['quarter'] = merged['day'].apply(get_academic_quarter)
    return merged

merged_data = prepare_merged_dataset(sensing_data, demographics_data, ema_data)

# Function to select features
def select_features(platform, category_list):
    features = []
    for category in category_list:
        if isinstance(feature_dict[category], dict):
            if platform in feature_dict[category]:
                features.extend(feature_dict[category][platform])
            if 'all' in feature_dict[category]:
                features.extend(feature_dict[category]['all'])
        else:
            features.extend(feature_dict[category])
    return features

# Identify columns with categorical data (e.g., 'gender', 'race')
categorical_columns = ['gender', 'race']  # Add more columns as necessary

# Build the pipeline with preprocessing for categorical data
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor


# Modify build_pipeline to handle both classifiers and regressors
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, mean_absolute_error, r2_score


def build_pipeline(merged_data, platform, categories, outcome, model_type="RandomForest"):
    features = select_features(platform, categories)
    missing_columns = [col for col in features if col not in merged_data.columns]
    if missing_columns:
        print(f"Missing columns: {missing_columns}")
        return None

    X = merged_data[features]
    y = merged_data[outcome]

    # Preprocessing pipeline for categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), categorical_columns)
        ],
        remainder='passthrough'
    )

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Choose the model
    if model_type == "RandomForest":
        model = RandomForestRegressor() if y.dtype.kind in 'fc' else RandomForestClassifier()
    elif model_type == "GradientBoosting":
        model = GradientBoostingRegressor() if y.dtype.kind in 'fc' else GradientBoostingClassifier()
    elif model_type == "AdaBoost":
        model = AdaBoostRegressor() if y.dtype.kind in 'fc' else AdaBoostClassifier()
    else:
        raise ValueError("Unsupported model_type. Choose 'RandomForest', 'GradientBoosting', or 'AdaBoost'.")

    # Build the pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('scaler', StandardScaler()),
        ('model', model)
    ])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Predict on the test set
    y_pred = pipeline.predict(X_test)

    # Classification Metrics
    if y.dtype.kind in 'bui':  # Integer or categorical target
        print(f"\n{model_type} Model Classification Metrics")
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.2f}")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:\n", cm)

        # Classification report
        print("\nClassification Report:\n", classification_report(y_test, y_pred))

        # Heatmap for confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'{model_type} Confusion Matrix Heatmap')
        plt.show()

    # Regression Metrics
    elif y.dtype.kind in 'fc':  # Continuous target
        print(f"\n{model_type} Model Regression Metrics")
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"Mean Squared Error (MSE): {mse:.2f}")
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"R-squared (R²): {r2:.2f}")

    return pipeline


# Test the updated function
pipeline = build_pipeline(merged_data, platform='ios', categories=['demographics', 'activity'],
                          outcome='sleep_duration', model_type="GradientBoosting")

pipeline = build_pipeline(merged_data, platform='ios', categories=['demographics', 'activity'],
                          outcome='sleep_duration', model_type="AdaBoost")
pipeline = build_pipeline(merged_data, platform='ios', categories=['demographics', 'activity'],
                          outcome='sleep_duration', model_type="RandomForest")
