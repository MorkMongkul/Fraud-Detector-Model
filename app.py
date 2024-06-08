import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import streamlit as st
import kaggle
import json

# Kaggle dataset configuration
dataset = 'mlg-ulb/creditcardfraud'
data_file = 'data/creditcard.csv'

# Function to download dataset from Kaggle
@st.cache(allow_output_mutation=True)
def download_data():
    if not os.path.exists('data'):
        os.makedirs('data')
    kaggle.api.dataset_download_files(dataset, path='data/', unzip=True)
    return pd.read_csv(data_file)

# Set up Kaggle API credentials
kaggle_config = {
    "username": os.environ.get('KAGGLE_USERNAME'),
    "key": os.environ.get('KAGGLE_KEY')
}

# Write Kaggle API credentials to file
with open(os.path.expanduser("~/.kaggle/kaggle.json"), "w") as f:
    json.dump(kaggle_config, f)

os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)

# Load the dataset
data = download_data()

# Preprocess the data
X = data.drop(columns=['Class'])
Y = data['Class']
smot = SMOTE(random_state=42)
X_res, Y_res = smot.fit_resample(X, Y)
X_train, X_test, Y_train, Y_test = train_test_split(X_res, Y_res, test_size=0.2, random_state=42)

# Train the model
xgb_model = XGBClassifier(n_estimators=50, max_depth=3, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, Y_train)

# Calculate accuracy for XGBoost
xgb_predictions = xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(Y_test, xgb_predictions)
print(f"XGBoost Accuracy: {xgb_accuracy}")

# Streamlit app
st.title('Credit Card Fraud Detection')
st.write('Please input all required features with comma-separated values')

# Input from user
input_features = st.text_input('Enter the features (comma-separated)')

# Submit button
submit = st.button('Submit')

if submit:
    try:
        features = np.array(input_features.split(','), dtype=np.float64).reshape(1, -1)
        if features.shape[1] != X_train.shape[1]:
            st.error(f"Expected {X_train.shape[1]} features, but got {features.shape[1]}")
        else:
            prediction = xgb_model.predict(features)

            if prediction[0] == 0:
                st.success('The transaction is legitimate!')
            else:
                st.error('The transaction is a fraud!')
    except ValueError:
        st.error("Please enter valid numeric values separated by commas.")
