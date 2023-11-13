import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

# Read the dataset
df = pd.read_csv('canada_per_capita_income.csv')

# Extract features (year) and target variable (income)
year = df[['year']].values
income = df['per capita income (US$)']

# Fit linear regression model
reg = linear_model.LinearRegression(fit_intercept=False)
reg.fit(year, income)

# Generate predictions for the original dataset
predictions = reg.predict(year)

# Plot the original data and the linear regression line
plt.scatter(year, income, color='black', label='Original Data')
plt.plot(year, predictions, color='blue', linewidth=2, label='Linear Regression')

plt.xlabel('Year')
plt.ylabel('Per Capita Income (US$)')
plt.legend()
plt.title('Linear Regression on Per Capita Income Over Years')
plt.show()
