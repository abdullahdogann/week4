# Importing the libraries
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

df = pd.read_csv('cars.csv')

df.drop('Unnamed: 0', axis = 1, inplace = True)

cars = df.select_dtypes(exclude=['object']).copy()
lm = LinearRegression()
X, y = cars.drop('price', axis=1), cars['price']

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(X, y)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2014, 220000, 3000,110]]))