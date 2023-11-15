import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sn

# data

digits = load_digits()
df = pd.DataFrame(digits.data)
df['target'] = digits.target

X = df.drop('target',axis='columns')
y = df.target
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2)

print(X_train.shape)

# model
model = RandomForestClassifier()
model.fit(X_train,y_train)
print(model.score(X_test,y_test))

#visualize error 
y_predicted = model.predict(X_test)
cm = confusion_matrix(y_test,y_predicted)
sn.heatmap(cm, annot=True)

plt.show()