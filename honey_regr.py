import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

# Load the honey production dataset
df = pd.read_csv("https://content.codecademy.com/programs/data-science-path/linear_regression/honeyproduction.csv")

# Step 1: Check the structure of the DataFrame
print(df.head())

# Step 2: Group the dataset by year and calculate the mean total production per year
prod_per_year = df.groupby('year').totalprod.mean()

# Step 3: Create X as the years and reshape it
X = prod_per_year.index.values.reshape(-1, 1)

# Step 4: Create y as the total production
y = prod_per_year.values

# Step 5: Scatter plot of y vs X
plt.scatter(X, y)
plt.xlabel('Year')
plt.ylabel('Total Honey Production')
plt.title('Honey Production Over Time')
plt.show()

# Step 6: Create and fit a linear regression model
regr = linear_model.LinearRegression()
regr.fit(X, y)

# Step 7: Print the slope and intercept of the model
print('Slope:', regr.coef_)
print('Intercept:', regr.intercept_)

# Step 8: Predictions
y_predict = regr.predict(X)

# Step 9: Plot the regression line
plt.scatter(X, y)
plt.plot(X, y_predict, color='red')
plt.xlabel('Year')
plt.ylabel('Total Honey Production')
plt.title('Honey Production Over Time with Linear Regression')
plt.show()

# Step 10: Predict future honey production for the year 2050
X_future = np.array(range(2013, 2051)).reshape(-1, 1)
future_predict = regr.predict(X_future)

# Step 11: Plot the predicted future honey production
plt.plot(X_future, future_predict, color='green')
plt.xlabel('Year')
plt.ylabel('Predicted Total Honey Production')
plt.title('Predicted Honey Production for the Year 2050')
plt.show()

# Step 12: Print the predicted honey production for the year 2050
print('Predicted honey production for the year 2050:', future_predict[-1])
