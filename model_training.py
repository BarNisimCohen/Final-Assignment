import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error

# Load the data into a pandas DataFrame
file_name = "output_all_students_Train_v10.xlsx"
data = pd.read_excel(file_name)
from madlan_data_prep import prepare_data
df = prepare_data(data)

# Separate the features (X) and target variable (y)
X = df.drop('price', axis=1)
y = df['price']

selected_columns = ['City-enc', 'type-enc', 'condition-enc', 'entrance_date-enc', 'hasElevator',
                    'hasParking', 'hasStorage', 'hasBalcony', 'hasMamad', 'furniture', 'room_number',
                    'Area', 'floor', 'total_floors']
X = X[selected_columns]

# Split the preprocessed data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

# Define the parameter grid for grid search
param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0],
              'l1_ratio': [0.2, 0.4, 0.6, 0.8]}

# Create an instance of the Elastic Net model
model =ElasticNet()

# Perform grid search using cross-validation
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Retrieve the best hyperparameters and model
best_alpha = grid_search.best_params_['alpha']
best_l1_ratio = grid_search.best_params_['l1_ratio']
best_model = grid_search.best_estimator_

# Evaluate the best model on the testing set
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print('Mean Squared Error:', mse)
print("RMSE:", rmse)

import pickle
pickle.dump(best_model, open("trained_model.pkl", "wb"))