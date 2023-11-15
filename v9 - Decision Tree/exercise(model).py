#Drops from the dataset : Ticket , parch , SibSp , name , Passenger ID

from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
import seaborn as sn

df = pd.read_csv('./titanic.csv')

# clean data and process them 

df = df.drop(['Name','PassengerId','SibSp','Parch','Ticket'],axis=1)

features = df.drop(['Survived'],axis=1)
target = df['Survived']

#print(target.head())

le_sex = LabelEncoder()
le_cabin = LabelEncoder()
le_embarked = LabelEncoder()

features['Sex_n'] = le_sex.fit_transform(features['Sex'])
features['Cabin_n'] = le_cabin.fit_transform(features['Cabin'])
features['Embarked_n'] = le_embarked.fit_transform(features['Embarked'])

features = features.drop(['Sex','Cabin','Embarked'],axis=1)

#print(features.head())

# model training
X_train , X_test , y_train , y_test = train_test_split(features , target ,test_size=0.2)
print(X_train.shape , y_test.shape)

model = tree.DecisionTreeClassifier()
model.fit(X_train,y_train)
print(model.score(X_test,y_test))

# see the failure
y_predicted = model.predict(X_test)
cm = confusion_matrix(y_test,y_predicted)
plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.xlabel('Truth')

plt.show()