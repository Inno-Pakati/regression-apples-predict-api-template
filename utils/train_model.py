"""
    Simple file to create a Sklearn model for deployment in our API

    Author: Explore Data Science Academy

    Description: This script is responsible for training a simple linear
    regression model which is used within the API for initial demonstration
    purposes.

"""

# Dependencies
from typing import Container
import pandas as pd
import pickle
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# Fetch training data and preprocess for modeling
train = pd.read_csv('data/train_data.csv')

df_train = train[(train['Commodities'] == 'APPLE GOLDEN DELICIOUS')].drop(['Date', 'Commodities', 'Container', 'Size_Grade', 'Province'], axis=1)

X = df_train.drop('avg_price_per_kg', axis=1)
y = df_train['avg_price_per_kg']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Fit model
reg_tree = DecisionTreeRegressor(max_depth=11, min_samples_leaf=7, random_state=5)
print ("Training Model...")
reg_tree.fit(X_train, y_train)

# Pickle model for use within our API
save_path = '../assets/trained-models/regr_tree.pkl'
print (f"Training completed. Saving model to: {save_path}")
pickle.dump(reg_tree, open(save_path,'wb'))
