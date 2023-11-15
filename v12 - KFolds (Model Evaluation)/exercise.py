from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression , LinearRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

iris = load_iris()
iterations = 2

lr = (cross_val_score(LogisticRegression(max_iter=10000),iris.data,iris.target,cv=iterations))
linReg = (cross_val_score(LinearRegression(),iris.data,iris.target,cv=iterations))
svc = (cross_val_score(SVC(),iris.data,iris.target,cv=iterations))
dt = (cross_val_score(DecisionTreeClassifier(),iris.data,iris.target,cv=iterations))
rf = (cross_val_score(RandomForestClassifier(),iris.data,iris.target,cv=iterations))

print('Logistic Regression : ', np.mean(lr))
print('Linear Regression : ', np.mean(linReg))
print('SVM : ', np.mean(svc))
print('Decision Tree : ', np.mean(dt))
print('Random Forest : ', np.mean(rf))
