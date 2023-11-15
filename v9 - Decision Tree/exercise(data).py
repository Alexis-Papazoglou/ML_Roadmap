#Drops from the dataset : Ticket , parch , SibSp , name , Passenger ID
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

df = pd.read_csv('./titanic.csv')
print(df.head())

# -- -- CHECK FOR DATA PATTERNS -- -- 

#check cabin impact
cabin_owner = df[df['Cabin'].notna()]
#print('Cabin count : ' , cabin_owner.shape[0]/df.shape[0])

plt.figure(figsize=(8,6))
sns.countplot(x='Survived',hue=df['Cabin'].isnull(),data=df,palette='pastel')
plt.title('Survivability by cabin')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.legend(title='Cabin Presence', labels=['Without Cabin', 'With Cabin'])

# check sex impact
female = 0
for sex in df['Sex']:
    if sex == 'female':
        female += 1
        
#print('Female count : ' , female/df.shape[0])

plt.figure(figsize=(8, 6))
sns.countplot(x='Survived', hue='Sex', data=df, palette='pastel')
plt.title('Survivability by Sex')
plt.xlabel('Survived')
plt.ylabel('Count')

#check fare impact

plt.figure(figsize=(12, 6))
sns.boxplot(x='Survived', y='Fare', data=df, palette='viridis', showfliers=False)  # Set showfliers to False to exclude outliers

# Customize the plot
plt.title('Fare Impact on Survivability')
plt.xlabel('Survived')
plt.ylabel('Fare')

# check parch impact

plt.figure(figsize=(8, 6))
sns.countplot(x='Survived', hue='Pclass', data=df, palette='pastel')
plt.title('Survivability by Pclass')
plt.xlabel('Survived')
plt.ylabel('Count')


plt.tight_layout()
plt.show()