import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

df = pd.read_csv('homeprices.csv')
le = LabelEncoder()

# using label encoder to handle string values
df.town = le.fit_transform(df.town)

# creating model inputs
X = df[['town' , 'area']].values
y = df.price.values

# one hot encoding the X 
ohe = ColumnTransformer(transformers=[('onehot' , OneHotEncoder() , [0])], remainder='passthrough')
X = ohe.fit_transform(X)

#droping one column
X = X[:,1:]

# train
model = LinearRegression()
model.fit(X,y)

#print(model.coef_ , model.intercept_)
print(model.predict([[1,0,2800]]))

print(model.score(X,y))