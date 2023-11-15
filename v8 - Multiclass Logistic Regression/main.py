import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sn
import numpy as np

# data preprocessing
digits = load_digits()

X_train , X_test , y_train , y_test = train_test_split(digits.data , digits.target , test_size=0.2)
print(X_train.shape , X_test.shape)

# model training

model = LogisticRegression(max_iter=10000)
model.fit(X_train,y_train)

# results

print(model.score(X_test,y_test))

# generate random numbers and test

np.random.seed(np.random.randint(0,100))
random_numbers = np.random.randint(0, 1797, size=5)

predictions_array , target_array = [] , []
for number in random_numbers:
    target_array.append(digits.target[number])
    predictions_array.append(model.predict([digits.data[number]]))

print('Actual : ' , target_array , 'Predictions : ' , np.array(predictions_array).flatten())

# see the faliure

y_predicted = model.predict(X_test)
cm = confusion_matrix(y_test,y_predicted)
plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.xlabel('Truth')

plt.show()