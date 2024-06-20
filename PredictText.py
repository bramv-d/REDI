import joblib
import pandas as pd


def prepare_text_features(file_path):
    data = pd.read_csv(file_path, index_col=False)
    features_importances = pd.read_csv('RegressionModels/ExtraTreesRegressor/run1/feature_importances.csv',delimiter='\t', index_col=False)

    features = features_importances['Feature'].to_numpy()
    data = data.dropna(axis=1)
    data = data.drop(columns=['Inputfile'])
    filtered_data = data[features]

    return filtered_data

def scale_data(text_features):
    scaler =  joblib.load('RegressionModels/ExtraTreesRegressor/run1/scaler.pkl')
    scaled_data = scaler.fit_transform(text_features)
    return scaled_data

def predict_text(text_features):
    difficulty_model = joblib.load('RegressionModels/ExtraTreesRegressor/run1/difficulty_model.pkl')
    predictions = difficulty_model.predict(text_features)
    print(predictions)


data = prepare_text_features("1boekje total.csv")
scaled_data = scale_data(data)
predict_text(scaled_data)