import csv
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, classification_report
import joblib
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import RFECV

def prepare_data(file_path, columns_to_keep=None):
    # Load the data into a pandas dataframe
    data = pd.read_csv(file_path, index_col=False)
    # Place the AVI score in a separate variable
    # Currently it is in the title column and we need to extract it
    first_column = data['Inputfile']
    # Extract AVI scores from strings
    avi_scores = [s.split('_')[1] for s in first_column]
    # Add the AVI scores to the dataframe
    data.insert(1, 'AVI', avi_scores)
    data = data.drop(columns=['Inputfile'])
    if columns_to_keep:
        return data[columns_to_keep]
    else: 
        return data

# Define a function to standardize features, train the model, and evaluate it
def train_and_evaluate_model(data, target_column, run_name, model_name):    
    # Separate features and target variable
    X = data.drop(columns=[target_column])
    X.dropna(axis=1, inplace=True)  # Drop columns with missing values
    X = X.select_dtypes(include=[np.number])

    y = data[target_column]
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Initialize the model with hyperparameter tuning using GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Initialize the model
    model = RandomForestRegressor(random_state=42)
    
    # Perform a systematic search over a grid of hyperparameters
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    if best_model.feature_importances_:
    # Feature selection using RFECV
        rfecv = RFECV(estimator=best_model, 
                    step=1, 
                    cv=5, scoring='neg_mean_squared_error')
        rfecv.fit(X_train, y_train)

        # Get the selected features
        selected_features = X.columns[rfecv.support_]
        # Train the model with the selected features
        X_train_selected = rfecv.transform(X_train)
        X_test_selected = rfecv.transform(X_test)
        best_model.fit(X_train_selected, y_train)
    else:
        best_model.fit(X_train, y_train)
    

    # Define the directory path
    directory_path = model_name + '/' + run_name

    # Create the directory if it does not exist
    os.makedirs(directory_path, exist_ok=True)

    # Convert the dictionary to a DataFrame
    best_params_df = pd.DataFrame([grid_search.best_params_])

    # Save the DataFrame to a CSV file
    best_params_df.to_csv(directory_path + '/best_params.csv', sep='\t', index=False)
    
    # Implement k-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    if rfecv:
        cv_scores = cross_val_score(best_model, X_scaled[:, rfecv.support_], y, cv=kf, scoring='neg_mean_squared_error')
    else:
        cv_scores = cross_val_score(best_model, X_scaled, y, cv=kf, scoring='neg_mean_squared_error')

    MSE_scores = -cv_scores
    avarage_MSE = -cv_scores.mean()
    
    # Custom scoring function to evaluate with tolerance
    def score_with_tolerance(y_true, y_pred, tolerance=1):
        within_tolerance = (abs(y_true - y_pred) <= tolerance).sum()
        return within_tolerance / len(y_true)
    
    # Save the model and scaler
    joblib.dump(best_model, model_name + '/' + run_name + '/difficulty_model.pkl')
    joblib.dump(scaler, model_name+ '/' + run_name + '/scaler.pkl')
    if rfecv:
        joblib.dump(rfecv, model_name+ '/' + run_name + '/rfecv.pkl')

    print("Model, scaler, and RFECV saved successfully.")

    # Make predictions
    y_pred = best_model.predict(X_test_selected)
    
    # Evaluate with tolerance
    tolerance = 1
    score = score_with_tolerance(y_test, y_pred, tolerance)
    # Write the score and average MSE to the CSV file
    file_name = 'scores.csv'
    file_path = os.path.join(directory_path, file_name)

    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Score","MSE scores", "Average MSE"])
        writer.writerow([score,MSE_scores,avarage_MSE ])

    # Train the model on the entire dataset to get feature importances
    best_model.fit(X_scaled[:, rfecv.support_], y)
    
    # Display feature importances
    feature_importances = best_model.feature_importances_
    feature_names = selected_features
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    
    print(importance_df)
    importance_df.to_csv(model_name+ '/' + run_name +  '/feature_importances.csv', sep='\t', index=False)
    
    return best_model, avarage_MSE

data = prepare_data("T-scan results/14-06-2024.csv")
target_column = 'AVI'
model, avg_mse = train_and_evaluate_model(data, target_column, 'run1', 'RandomForestRegressor')
