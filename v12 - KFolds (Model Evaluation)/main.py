from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import numpy as np

digits = load_digits()

lr = (cross_val_score(LogisticRegression(max_iter=10000),digits.data,digits.target,cv=10))
svc = (cross_val_score(SVC(),digits.data,digits.target,cv=10))
rf = (cross_val_score(RandomForestClassifier(),digits.data,digits.target,cv=10))

print('Logistic Regression : ', np.mean(lr))
print('SVM : ', np.mean(svc))
print('Random Forest : ', np.mean(rf))
