import joblib
import pandas as pd
import numpy as np

path_to_run = 'RegressionModels/ExtraTreesRegressor/run4/'

def prepare_text_features(data):
    # Load the input data

    # Load feature importances and ensure it contains the feature names in the correct order
    features_importances = pd.read_csv(path_to_run + 'feature_importances.csv', delimiter='\t', index_col=False)
    features = features_importances['Feature'].to_numpy()

    # Drop columns with NaN values and the 'Inputfile' column
    data = data.dropna(axis=1)
    if 'Inputfile' in data.columns:
        data = data.drop(columns=['Inputfile'])

    # Ensure the data has all required features and in the correct order
    # Filter the data to keep only the required features, preserving their original order
    filtered_data = data[[feature for feature in data.columns if feature in features]]
    
    return filtered_data

def scale_data(text_features):
    # Load the pre-trained scaler
    scaler = joblib.load(path_to_run + 'scaler.pkl')
    # Use the scaler to transform the text features
    scaled_data = scaler.transform(text_features)
    return scaled_data

def predict_text(text_features):
    # Load the pre-trained model
    difficulty_model = joblib.load(path_to_run + 'difficulty_model.pkl')
    
    # Predict using the transformed text features
    predictions = difficulty_model.predict(text_features)
    return predictions

def visualise_data(predictions, raw_data):
    first_column = raw_data['Inputfile']
    # Extract AVI scores from strings
    avi_scores = [int(s.split('_')[1]) for s in first_column]
    # Convert the list to a NumPy array
    avi_scores_array = np.array(avi_scores, dtype=int)
    # Add the AVI scores to the dataframe

    raw_data.insert(1, 'Original AVI', avi_scores_array)    
    raw_data.insert(2, 'Predicted AVI', predictions)
    rounded_AVI = (np.rint(np.round(predictions))).astype(int)
    raw_data.insert(3, 'Rounded AVI', rounded_AVI)
    raw_data.insert(4, 'Correct', rounded_AVI == avi_scores)
    columns_to_keep = ['Original AVI', 'Predicted AVI', 'Rounded AVI', 'Correct']
    raw_data = raw_data[columns_to_keep]
    print(raw_data)


# Prepare and transform the input data
raw_data = pd.read_csv("Holdout sample/predict.csv", index_col=False)

data = prepare_text_features(raw_data)
scaled_data = scale_data(data)

# Make predictions
predictions = predict_text(scaled_data)
visualise_data(predictions, raw_data)
