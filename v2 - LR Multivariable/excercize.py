import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv('hiring.csv')

# Correct the data

# replacing numerical with numbers
df.experience = df.experience.fillna("zero")

experience_mapping = { 'zero': 0 ,'one': 1,'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6,
                       'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11,
                       'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
                       'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20}

df['experience'] = df['experience'].map(experience_mapping)

#fixing the NaN data
df['test_score(out of 10)'].fillna(np.floor(df['test_score(out of 10)'].mean()), inplace=True)
    #print(df)

#Training the model
reg = linear_model.LinearRegression()
reg.fit(df[['experience','test_score(out of 10)','interview_score(out of 10)']],df['salary($)'])
    #print(reg.coef_,reg.intercept_)

#making predictions
print(reg.predict([[2,9,6]]))
print(reg.predict([[12,10,10]]))
