import pandas as pd
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Preparing data
iris = load_iris()
df = pd.DataFrame(iris.data,columns=iris.feature_names)
df['target'] = iris.target

df0 = df[df.target==0]
df1 = df[df.target==1]
df2 = df[df.target==2]

plt.scatter(df1['sepal length (cm)'],df1['sepal length (cm)'] , c='green' , marker='+')
plt.scatter(df2['sepal length (cm)'],df2['sepal length (cm)'] , c='blue', marker='.')

#plt.show()

X = df.drop(['target'],axis=1)
y = df.target

X_train , X_test , y_train , y_test = train_test_split(X , y , train_size=0.8)

# model 
model = SVC()
model.fit(X_train,y_train)
print(model.score(X_test,y_test))