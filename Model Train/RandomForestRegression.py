import csv
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, classification_report
import joblib
from sklearn.tree import DecisionTreeRegressor

# Define a function to filter the columns
def filter_columns(data, columns_to_keep):
    return data[columns_to_keep]

# Define a function to standardize features, train the model, and evaluate it
def train_and_evaluate_model(data, target_column, columns_to_keep, run_name, model_name):
    # Filter the data to keep only the desired columns

    filtered_data = filter_columns(data, columns_to_keep)
    
    # Separate features and target variable
    X = filtered_data.drop(columns=[target_column])
    X.dropna(axis=1, inplace=True)  # Drop columns with missing values
    X = X.select_dtypes(include=[np.number])

    y = filtered_data[target_column]
    
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
    # model = DecisionTreeRegressor(random_state=42)
    
    # perform a systematic search over a grid of hyperparameters
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

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
    cv_scores = cross_val_score(best_model, X_scaled, y, cv=kf, scoring='neg_mean_squared_error')
    
    MSE_scores = -cv_scores
    avarage_MSE = -cv_scores.mean()
    
    # Custom scoring function to evaluate with tolerance
    def score_with_tolerance(y_true, y_pred, tolerance=1):
        within_tolerance = (abs(y_true - y_pred) <= tolerance).sum()
        return within_tolerance / len(y_true)
    
    # Train the model on the entire dataset
    best_model.fit(X_train, y_train)

    # Save the model and scaler
    joblib.dump(best_model, model_name + '/' + run_name + '/difficulty_model.pkl')
    joblib.dump(scaler, model_name+ '/' + run_name + '/scaler.pkl')

    print("Model and scaler saved successfully.")

    # Make predictions
    y_pred = best_model.predict(X_test)
    
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
    best_model.fit(X_scaled, y)
    
    # Display feature importances
    feature_importances = best_model.feature_importances_
    feature_names = X.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    
    print(importance_df)
    importance_df.to_csv( model_name+ '/' + run_name +  '/feature_importances.csv', sep='\t', index=False)
    
    
    
    return best_model, -cv_scores.mean()

# Load the data
data = pd.read_csv('./results_with_avi.csv')
dropdata = data.dropna(axis=1)
columnames = dropdata.columns.to_list()
target_column = 'AVI'
# columns_to_keep = columnames
# Keep these columns based on related work
columns_to_keep = ['Let_per_wrd_corr', 'Props_dz_tot', 'AL_gem', 'AVI']  # Another variation

# columns_to_keep = [
#     "Props_dz_tot",
#     "AL_gem",
#     "Wrd_per_let",
#     "Wrd_per_let_zn",
#     "Let_per_wrd_corr",
#     "Let_per_wrd",
#     "Wrd_per_dz",
#     "Let_per_wrd_zn",
#     "Dzin_per_wrd",
#     "Bijv_bep_dz",
#     "AL_max",
#     "Let_per_wrd_nw",
#     "Pv_Frog_d",
#     "Wrd_per_zin",
#     "AL_sub_ww",
#     "Wrd_per_nwg",
#     "Vg_d",
#     "Let_per_wrd_nw_corr",
#     "Zin_per_wrd",
#     "Wrd_per_morf_zn",
#     "Inhwrd_dz_zonder_abw",
#     "Hzin_conj",
#     "Lem_freq_log_zonder_abw",
#     "Pv_Frog_per_zin",
#     "Interp_d",
#     "Bijw_d",
#     "Bijv_bep_d",
#     "Fin_bijzin_per_zin",
#     "Let_per_wrd_nsam",
#     "TTR_lem",
#     "Morf_per_wrd_zn",
#     "LiNT_score2",
#     "Bijzin_conj",
#     "MTLD_inhwrd_zonder_abw",
#     "Conc_nw_strikt_p",
#     "Pv_Alpino_per_zin",
#     "Conc_ww_p",
#     "TTR_inhwrd_zonder_abw",
#     "AVI"
# ]
model, avg_mse = train_and_evaluate_model(data, target_column, columns_to_keep, 'run3', 'RandomForestRegressor')



# # Keep these columns based on related work
# columns_to_keep_2 = ['Wrd_per_zin','Pers_vnw_d', "AL_max", "Let_per_wrd_corr", 'Props_dz_tot', 'AL_gem', 'AVI']  # Include the target column in the list
# model_2, avg_mse_2 = train_and_evaluate_model(data, target_column, columns_to_keep_2)

# # Keep these columns based on ML Feature findings
# columns_to_keep_3 = ['Let_per_wrd_corr', 'Props_dz_tot', 'AL_gem', 'AVI']  # Another variation
# model_3, avg_mse_3 = train_and_evaluate_model(data, target_column, columns_to_keep_3)