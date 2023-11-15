import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import tree

# data preprocessing
df = pd.read_csv('./salaries.csv')

features = df.drop('salary_more_then_100k' , axis=1)
target = df['salary_more_then_100k']

le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()

features['company_n'] = le_company.fit_transform(features['company'])
features['job_n'] = le_company.fit_transform(features['job'])
features['degree_n'] = le_company.fit_transform(features['degree'])

features = features.drop(['job','company','degree'],axis=1)
print(features.head())

X_train , X_test , y_train , y_test = train_test_split(features , target , test_size=0.2)

# model training
model = tree.DecisionTreeClassifier()
model.fit(X_train,y_train)
print(model.score(X_test,y_test))