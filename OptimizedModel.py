import csv
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, classification_report
import joblib
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import RFE, RFECV

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
    data = data.dropna(axis=1)
    data = data.drop(columns=['Inputfile'])
    # Optionally select columns to keep, including 'AVI'
    if columns_to_keep is not None:
        columns_to_keep.append('AVI')  # Ensure 'AVI' is included
        return data[columns_to_keep]
    else:
        return data


# Define a function to standardize features, train the model, and evaluate it
def train_and_evaluate_model(data, target_column, run_name, model_name):    
    
    # Initialize the model
    model = ExtraTreesRegressor(random_state=42)

    # Separate features and target variable
    X = data.drop(columns=[target_column])
    # X.dropna(axis=1, inplace=True)  # Drop columns with missing values
    X = X.select_dtypes(include=[np.number])

    y = data[target_column]

    # Initialize RFE with the model and number of features to select
    rfe = RFE(model, n_features_to_select=10)  # Adjust n_features_to_select based on your needs
    rfe.fit(X, y)

    # Get the selected features
    selected_features = X.columns[rfe.support_]
    X_selected = X[selected_features]
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Initialize the model with hyperparameter tuning using GridSearchCV
    param_grid = {
        'n_estimators': [100],
        'max_depth': [None],
        'min_samples_split': [2],
        'min_samples_leaf': [4]
    }
    
    # Perform a systematic search over a grid of hyperparameters
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    best_model.fit(X_train, y_train)
    

    # Define the directory path
    directory_path = 'RegressionModels/' + model_name + '/' + run_name

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
    
    # Save the model and scaler
    joblib.dump(best_model, directory_path + '/difficulty_model.pkl')
    joblib.dump(scaler, directory_path + '/scaler.pkl')

    # Write the score and average MSE to the CSV file
    file_path = os.path.join(directory_path, 'scores.csv')

    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["MSE scores", "Average MSE"])
        writer.writerow([MSE_scores,avarage_MSE ])

    best_model.fit(X_scaled, y)
    # Display feature importances
    feature_importances = best_model.feature_importances_
    feature_names = X_selected.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    
    print(importance_df)
    importance_df.to_csv(directory_path +  '/feature_importances.csv', sep='\t', index=False)
    return best_model, avarage_MSE

Columns_to_keep = ["Let_per_wrd_corr","Let_per_wrd_zn","Let_per_wrd","Wrd_per_let","AL_gem","Props_dz_tot","Wrd_per_let_zn","Wrd_per_zin","Wrd_per_morf","AL_max","Wrd_per_morf_zn","Wrd_per_dz","Zin_per_wrd","Morf_per_wrd_zn","Morf_per_wrd","Dzin_per_wrd","Let_per_wrd_nsam","Wrd_per_nwg","Let_per_wrd_nw_corr","AL_sub_ww","Fin_bijzin_per_zin","Bijzin_per_zin"]
Columns_related_work = ["wrd_freq_log_zn_corr", "wrd_freq_zn_log", "Conc_nw_ruim_p", "Conc_nw_strikt_p", "Alg_nw_d", "Pers_ref_d", "Pers_vnw_d", "Wrd_per_zin", "Wrd_per_dz", "Inhwrd_dz_zonder_abw", "AL_max", "Bijzin_per_zin", "Bijv_bep_dz_zbijzin", "Extra_KConj_dz", "MTLD_inhwrd_zonder_abw"]
data = prepare_data("T-scan results/14-06-2024.csv")
target_column = 'AVI'

model, avg_mse = train_and_evaluate_model(data, target_column, 'run4', 'ExtraTreesRegressor')
