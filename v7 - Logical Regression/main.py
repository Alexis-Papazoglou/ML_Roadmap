import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('insurance_data.csv')

# spliting the data
X_train , X_test , y_train , y_test = train_test_split(df[['age']],df.bought_insurance,test_size=0.1)

# train the model
model = LogisticRegression()
model.fit(X_train,y_train)

# results
predictions = (model.predict(X_test))
print(model.score(X_test,y_test))
print(model.predict_log_proba(X_test))

# plot predictions vs. y_test
plt.scatter(X_test, y_test, color='red', label='Actual')
plt.scatter(X_test, predictions, color='blue', marker='+', label='Predicted')
plt.xlabel('Age')
plt.ylabel('Bought Insurance')
plt.title('Actual vs. Predicted')
plt.legend()
plt.show()