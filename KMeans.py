
# Iris data

# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# importing dataset
data_set=pd.read_csv("IRIS (1).csv")
df=pd.DataFrame(data_set)
print("Actual Dataset")
print(df.to_string())

# to get information
data_set.info()

# to drop null values
data_set.isna().sum()

# to drop duplicate values
print('no ofduplicates',data_set.duplicated().sum())
df=df.drop_duplicates()
print(df.shape)
print(df.info())

# Extract features
'''x = df.iloc[:,:-1].values
print(x)'''
x=df

# K-Means Clustering with the Elbow Method
from sklearn.cluster import KMeans
wcss_list= []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state= 42)
    kmeans.fit(x)
    wcss_list.append(kmeans.inertia_)
print(wcss_list)


# Plot the Elbow Method graph
plt.plot(range(1, 11), wcss_list)
plt.title('The Elobw Method Graph')
plt.xlabel('Number of clusters(k)')
plt.ylabel('wcss_list')
plt.show()


# K-Means Clustering with optimal k
kmeans = KMeans(n_clusters=3, init='k-means++', random_state= 42)
y_predict= kmeans.fit_predict(x)
print(y_predict)

# K-Means Clustering with different k
kmeans = KMeans(n_clusters=2, init='k-means++', random_state= 42)
y_predict= kmeans.fit_predict(x)
print(y_predict)
