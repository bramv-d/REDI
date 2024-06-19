import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Function to preprocess the new data
def preprocess_new_data(new_data, columns_to_keep, scaler):
    # Filter the new data to keep only the desired columns
    filtered_data = new_data[columns_to_keep]
    
    # Drop the target column 'AVI' as it's not available in the new data
    if 'AVI' in filtered_data.columns:
        filtered_data = filtered_data.drop(columns=['AVI'])
    
    # Standardize the features
    X_scaled = scaler.transform(filtered_data)
    
    return X_scaled

# Function to load the model and scaler
def load_model_and_scaler(model_path, scaler_path):
    # Load the saved model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    return model, scaler

# Function to predict AVI values for new data
def predict_new_data(new_data_file, model_path, scaler_path, columns_to_keep):
    # Load the new data
    new_data = pd.read_csv(new_data_file)
    print(new_data['AL_max'])
    # Load the model and scaler
    model, scaler = load_model_and_scaler(model_path, scaler_path)
    
    # Preprocess the new data
    X_scaled = preprocess_new_data(new_data, columns_to_keep, scaler)
    print(X_scaled)
    # # Make predictions
    # predictions = model.predict(X_scaled)
    
    # return predictions

# File paths and column definitions
new_data_file = '1boekje total.doc.csv'  # Replace with the actual path to your new data file
model_path = 'Model Train/difficulty_model.pkl'
scaler_path = 'Model Train/scaler.pkl'
columns_to_keep = ['Wrd_per_zin','Pers_vnw_d', "AL_max", "Let_per_wrd_corr", 'Props_dz_tot', 'AL_gem']  # The columns used during training

# Predict AVI values for the new data
predictions = predict_new_data(new_data_file, model_path, scaler_path, columns_to_keep)

# Print the predictions
print(predictions)
