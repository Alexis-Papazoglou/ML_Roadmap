import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import os

# Set LOKY_MAX_CPU_COUNT to the number of cores you want to use
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # Set to the desired number of cores

fig, axs = plt.subplots(1, 5, figsize=(20, 5))  # Creates 1 row and 3 columns of subplots

# data preprocessing 

iris = load_iris()
df = pd.DataFrame(iris.data,columns=iris.feature_names)
df.rename(columns={'petal length (cm)': 'Length', 'petal width (cm)': 'Width'}, inplace=True)
df = df.drop(['sepal length (cm)','sepal width (cm)'],axis=1)

axs[0].scatter(df.Width,df.Length)
axs[0].set_title('Unscaled Data')

# model training

km = KMeans(n_clusters=2,n_init=2)
y_predicted = km.fit_predict(df)

df['cluster'] = y_predicted

df0 = df[df['cluster']==0]
df1 = df[df['cluster']==1]

axs[1].scatter(df0.Width,df0.Length)
axs[1].scatter(df1.Width,df1.Length)
axs[1].scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],c='purple',marker='+')
axs[1].set_title('Unscaled Clustering')
# cluster centers are way off

# scale the data properly

scaler = MinMaxScaler()
scaler.fit(df[['Width']])
df['Width'] = scaler.transform(df[['Width']])
scaler.fit(df[['Length']])
df['Length'] = scaler.transform(df[['Length']])

axs[2].scatter(df.Width,df.Length)
axs[2].set_title('Scaled Data')

# model training again

km = KMeans(n_clusters=2,n_init=2)
y_predicted = km.fit_predict(df)

df['cluster'] = y_predicted

df0 = df[df['cluster']==0]
df1 = df[df['cluster']==1]

axs[3].scatter(df0.Width,df0.Length)
axs[3].scatter(df1.Width,df1.Length)
axs[3].scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],c='purple',marker='+')
axs[3].set_title('Scaled Clustering')

# we can see our cluster_centers are correct now

# Find Elbow

k_rng = range(1,5)
sse = []
for k in k_rng:
    km = KMeans(n_clusters=k , n_init=10)
    km.fit(df)
    sse.append(km.inertia_) 

axs[4].plot(k_rng,sse)
axs[4].set_title('Elbow')

plt.show()