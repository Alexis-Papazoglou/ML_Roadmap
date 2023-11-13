# what we will do is one hot encoding ,
# we will add 3 columns with each towns name and we will use 1 or 0 to specify if the building is in that town
# that way we will handle the string data in our dataset

import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv('homeprices.csv')

# creating the columns for the towns
dummies = (pd.get_dummies(df.town).astype(int))

#adding dummies to the df and removing town column
df = pd.concat([df,dummies],axis=1)
df = df.drop('town', axis=1)

#we also will remove one dummy variable column to avoid DUMMY VARIABLE TRAP
df = df.drop('west windsor',axis=1)

# Create train data
X = df.drop('price',axis=1).values
y = df.price.values

# Train the model
model = LinearRegression()
model.fit(X,y)

# See results
print(model.coef_ , model.intercept_)
print(model.predict([[2800,0,1]]))
print(model.predict([[3400,0,0]]))

# Check accuracy 
print(model.score(X,y))
