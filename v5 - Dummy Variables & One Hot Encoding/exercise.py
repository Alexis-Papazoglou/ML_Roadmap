import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv('carprices.csv')

# prepare data
dummies = (pd.get_dummies(df['Car Model']).astype(int))

df = pd.concat([df,dummies],axis=1)

df = df.drop(['Car Model','Mercedez Benz C class'],axis=1)
print(df)

X = df.drop('Sell Price($)',axis=1).values
y = df['Sell Price($)'].values

# train
model = LinearRegression()
model.fit(X,y)

# predict
print(model.predict([[45000,4,0,0]]))
print(model.predict([[86000,7,0,1]]))
print(model.score(X,y))