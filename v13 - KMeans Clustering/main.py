import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import os

# Set LOKY_MAX_CPU_COUNT to the number of cores you want to use
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # Set to the desired number of cores

# preprocess data

df = pd.read_csv('./income.csv')

df = df.drop(['Name'],axis=1)

fig, axs = plt.subplots(1, 4, figsize=(20, 5))  # Creates 1 row and 3 columns of subplots
axs[0].scatter(df.Age,df['Income($)'])

# k means model

km = KMeans(n_clusters=3,n_init=3)
y_predicted = km.fit_predict(df[['Age','Income($)']])

df['cluster'] = y_predicted

df0 = df[df.cluster == 0]
df1 = df[df.cluster == 1]
df2 = df[df.cluster == 2]

axs[1].scatter(df0.Age,df0['Income($)'],c='red')
axs[1].scatter(df1.Age,df1['Income($)'],c='blue')
axs[1].scatter(df2.Age,df2['Income($)'],c='green')
axs[1].scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],c='purple',marker='+')

# we can see that there is a problem with the clustering of the data
# this occurs because salary and age are scaled in different range
# we will use MinMaxScaler to fix it

scaler = MinMaxScaler()
scaler.fit(df[['Income($)']])
df[['Income($)']] = scaler.transform(df[['Income($)']])
scaler.fit(df[['Age']])
df[['Age']] = scaler.transform(df[['Age']])

# train model again

km = KMeans(n_clusters=3,n_init=3)
y_predicted = km.fit_predict(df[['Age','Income($)']])

df['cluster'] = y_predicted

df0 = df[df.cluster == 0]
df1 = df[df.cluster == 1]
df2 = df[df.cluster == 2]

axs[2].scatter(df0.Age,df0['Income($)'],c='red')
axs[2].scatter(df1.Age,df1['Income($)'],c='blue')
axs[2].scatter(df2.Age,df2['Income($)'],c='green')
axs[2].scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],c='purple',marker='+')

# we can see now the clustering is correct!

# Elbow plot method measurements

k_rng = range(1,10)
sse = []
for k in k_rng:
    km = KMeans(n_clusters=k , n_init=10)
    km.fit(df)
    sse.append(km.inertia_) # km.inertia gives the error of the model

axs[3].plot(k_rng,sse)

plt.show()

# we can see the elbow is on 3 so our assumption of using 