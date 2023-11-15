import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sn

# data preprocessing

iris = load_iris()
df = pd.DataFrame(iris.data)
df['target'] = iris.target
print(df.head())

X = df.drop(['target'],axis=1)
y = df['target']
X_train , X_test , y_train , y_test = train_test_split(X,y,train_size=0.2)

# model 

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train,y_train)
print(model.score(X_test,y_test))

y_predicted = model.predict(X_test)
cm = confusion_matrix(y_test,y_predicted)
sn.heatmap(cm, annot=True)

plt.show()