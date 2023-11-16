import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv('./HR_comma_sep.csv')
# encoding variables
department_dummies = pd.get_dummies(df['Department'], prefix='dept')
salary_dummies = pd.get_dummies(df['salary'], prefix='salary')
df_encoded = pd.concat([df, department_dummies, salary_dummies], axis=1)
df_encoded.drop(['Department', 'salary'], axis=1, inplace=True)
df_encoded.drop(['dept_support', 'salary_medium'], axis=1, inplace=True)

# Replace True and False with 1 and 0
df = df_encoded.replace({True: 1, False: 0})

# spliting the data
X_train , X_test , y_train , y_test = train_test_split(df.drop('left', axis=1),df['left'],test_size=0.3)

# model training
model = LogisticRegression(max_iter=100000)
model.fit(X_train,y_train)

# predictions 
predictions = (model.predict(X_test))
print(model.score(X_test,y_test))

print(predictions.shape)
print(y_test.shape)

# visualize results

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f'Model Accuracy: {accuracy:.2%}')

# Create a bar chart
plt.bar(['Correct', 'Incorrect'], [accuracy, 1 - accuracy], color=['green', 'red'])
plt.xlabel('Prediction Accuracy')
plt.ylabel('Proportion')
plt.title('Model Prediction Accuracy')
plt.show()