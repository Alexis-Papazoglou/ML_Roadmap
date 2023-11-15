import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sn
import numpy as np

# data
iris = load_iris()

X_train , X_test , y_train , y_test = train_test_split(iris.data , iris.target , train_size=0.2)

# model
model = LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)
print(model.score(X_test,y_test))

# testing
np.random.seed(np.random.randint(0,100))
random_numbers = np.random.randint(0, 150, size=5)
predictions_array , target_array = [] , []
for number in random_numbers:
    target_array.append(iris.target[number])
    predictions_array.append(model.predict([iris.data[number]]))

print('Actual : ' , target_array , 'Predictions : ' , np.array(predictions_array).flatten())

# visualise failure

y_predicted = model.predict(X_test)
cm = confusion_matrix(y_test,y_predicted)
plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True)
shape_text = f'Shape: {iris.data.shape}'
plt.text(0 , 0 , shape_text , ha='left')
plt.xlabel('Predicted')
plt.xlabel('Truth')

plt.show()