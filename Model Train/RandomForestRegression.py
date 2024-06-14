from matplotlib import pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, classification_report
import joblib

# Define a function to filter the columns
def filter_columns(data, columns_to_keep):
    return data[columns_to_keep]

# Define a function to standardize features, train the model, and evaluate it
def train_and_evaluate_model(data, target_column, columns_to_keep):
    # Filter the data to keep only the desired columns
    filtered_data = filter_columns(data, columns_to_keep)
    
    # Separate features and target variable
    X = filtered_data.drop(columns=[target_column])
    X.dropna(axis=1, inplace=True)  # Drop columns with missing values
    y = filtered_data[target_column]
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize the model
    model = RandomForestRegressor(random_state=42)
    
    # Implement k-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_scaled, y, cv=kf, scoring='neg_mean_squared_error')
    
    # Display the cross-validation scores
    print("Cross-validation MSE scores:", -cv_scores)
    print("Average MSE:", -cv_scores.mean())
    
    # Custom scoring function to evaluate with tolerance
    def score_with_tolerance(y_true, y_pred, tolerance=0.5):
        within_tolerance = (abs(y_true - y_pred) <= tolerance).sum()
        return within_tolerance / len(y_true)
    
    # Train the model on the entire dataset
    model.fit(X_scaled, y)

    # Save the model and scaler
    joblib.dump(model, 'Model Train/difficulty_model.pkl')
    joblib.dump(scaler, 'Model Train/scaler.pkl')

    print("Model and scaler saved successfully.")
    
    # Make predictions
    y_pred = model.predict(X_scaled)
    
    # Evaluate with tolerance
    score = score_with_tolerance(y, y_pred)
    
    print(f"Score with tolerance of ±1: {score * 100:.2f}%")
    
    # Train the model on the entire dataset to get feature importances
    model.fit(X_scaled, y)
    
    # Display feature importances
    feature_importances = model.feature_importances_
    feature_names = X.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    
    print(importance_df)

    
    
    return model, -cv_scores.mean()


# Load the data
data = pd.read_csv('./results_with_avi.csv')
target_column = 'AVI'

# Keep these columns and drop the rest
columns_to_keep = ['Wrd_per_zin','Pers_vnw_d', "AL_max", "Let_per_wrd_corr", 'Props_dz_tot', 'AL_gem', 'AVI']  # Include the target column in the list

# Train and evaluate the model with the specified columns
model, avg_mse = train_and_evaluate_model(data, target_column, columns_to_keep)

# You can now try different combinations of columns
columns_to_keep_2 = ['Let_per_wrd_corr', 'Props_dz_tot', 'AL_gem', 'AVI']  # Another variation
model_2, avg_mse_2 = train_and_evaluate_model(data, target_column, columns_to_keep_2)
