# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 17:47:39 2023

@author: berber
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score,KFold
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# Load the data into a pandas DataFrame
file_name = "output_all_students_Train_v10.xlsx"
data = pd.read_excel(file_name)
from madlan_data_prep import prepare_data
df=prepare_data(data)

# Separate the features (X) and target variable (y)
X = df.drop('price', axis=1)
y = df['price']

X['City'] = X['City'].astype(str)
X['type'] = X['type'].astype(str)
X['condition'] = X['condition'].astype(str)
X['city_area'] = X['city_area'].astype(str)
X['entrance_date'] = X['entrance_date'].astype(str)
# Select the relevant columns for the features and target variable
selected_columns = ['City', 'type', 'room_number', 'Area', 'hasElevator',
'hasParking', 'hasStorage','hasBalcony', 'hasMamad','condition','city_area','furniture','floor','total_floors','entrance_date']
X = X[selected_columns]
category_columns=['City', 'type','condition','city_area','entrance_date']
numeric_columns=['room_number', 'Area','hasElevator', 'hasParking', 'hasStorage','hasBalcony', 'hasMamad','furniture','floor','total_floors'] 
encoder = OrdinalEncoder()
scaler = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('ordinal_encoder', encoder, category_columns),
        ('scaler', scaler, numeric_columns)
    ], remainder='passthrough'
)
# Apply preprocessing to the data
X_preprocessed = preprocessor.fit_transform(X)

# Split the preprocessed data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=(422))

# Create an instance of the Elastic Net model
model = ElasticNet(alpha=2.95, l1_ratio=0.5,random_state=(422))

pipe_preprocessing_model = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model),
])

cv = KFold(n_splits=10, shuffle=True, random_state=(422))
cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')
mean_cv_score = cv_scores.mean()
std_cv_score = cv_scores.std()

print("Mean R2 Score:", mean_cv_score)
print("Standard Deviation of R2 Score:", std_cv_score)

# Train the Elastic Net model
model.fit(X_train, y_train)

# Evaluate the model on the testing set
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print('Mean Squared Error:', mse)
print("RMSE:", rmse)


import pickle
pickle.dump(model, open("trained_model.pkl","wb"))