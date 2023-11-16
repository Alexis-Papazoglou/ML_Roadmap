from matplotlib import pyplot as plt
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import seaborn as sns

# clean data and process them 

df = pd.read_csv('./titanic.csv')

df = df.drop(['Name','PassengerId','SibSp','Parch','Ticket'],axis=1)

features = df.drop(['Survived'],axis=1)
target = df['Survived']

features.Age = features.Age.fillna(features.Age.mean())

le_sex = LabelEncoder()
le_cabin = LabelEncoder()
le_embarked = LabelEncoder()

features['Sex_n'] = le_sex.fit_transform(features['Sex'])
features['Cabin_n'] = le_cabin.fit_transform(features['Cabin'])
features['Embarked_n'] = le_embarked.fit_transform(features['Embarked'])

features = features.drop(['Sex','Cabin','Embarked'],axis=1)

X_train , X_test , y_train , y_test = train_test_split(features,target,train_size=0.8)

# Train model

model = GaussianNB()
model.fit(X_train,y_train)
print(model.score(X_test,y_test))

y_predicted = model.predict(X_test)
cm = confusion_matrix(y_predicted,y_test)
sns.heatmap(cm,annot=True)
plt.show()