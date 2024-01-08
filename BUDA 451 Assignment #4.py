#!/usr/bin/env python
# coding: utf-8

# In[1]:


# read the orginal data
import pandas as pd
data = pd.read_csv("https://raw.githubusercontent.com/binbenliu/Teaching/main/data/USArrests.txt")
data


# In[2]:


# normalzied the data, please use this data for this question
from sklearn import preprocessing
df = data[['Murder','Assault','UrbanPop','Rape']]
scaled_df = pd.DataFrame(preprocessing.scale(df), index=data['State'], columns = df.columns)
scaled_df


# In[3]:


#4a

from sklearn import cluster

k_means = cluster.KMeans(n_clusters=3, max_iter=50, random_state=1)
k_means.fit(scaled_df)
labels = k_means.labels_
pd.DataFrame(labels, index=data.State, columns=['Cluster ID'])


# In[5]:


centroids = k_means.cluster_centers_
pd.DataFrame(centroids,columns = df.columns)


# In[7]:


#4b

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

numClusters = [1,2,3,4,5,6]
SSE = []
for k in numClusters:
    k_means = cluster.KMeans(n_clusters=k)
    k_means.fit(scaled_df)
    SSE.append(k_means.inertia_)

plt.plot(numClusters, SSE)
plt.xlabel('Number of Clusters')
plt.ylabel('SSE')


# In[8]:


#4c

#Single Link
import numpy as np
from matplotlib.pyplot import figure
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

def plot_dendrogram(model, names, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, labels=names.tolist(), **kwargs)

figure(figsize=(8, 6))

names = data['State']
X = data.drop(['State'],axis=1)

# setting distance_threshold=0 ensures we compute the full tree.
model = AgglomerativeClustering(distance_threshold=0, linkage='single', n_clusters=None)

model = model.fit(X)
plt.title("Hierarchical Clustering Dendrogram")
# plot the top three levels of the dendrogram
plot_dendrogram(model, names, truncate_mode="level", orientation='right')
plt.show()


# In[9]:


#Complete

figure(figsize=(8, 6))

# setting distance_threshold=0 ensures we compute the full tree.
model = AgglomerativeClustering(distance_threshold=0, linkage='complete', n_clusters=None)

model = model.fit(X)
plt.title("Hierarchical Clustering Dendrogram")
# plot the top three levels of the dendrogram
plot_dendrogram(model, names, truncate_mode="level", orientation='right')
plt.show()


# In[10]:


#Average

figure(figsize=(8, 6))

# setting distance_threshold=0 ensures we compute the full tree.
model = AgglomerativeClustering(distance_threshold=0, linkage='average', n_clusters=None)

model = model.fit(X)
plt.title("Hierarchical Clustering Dendrogram")
# plot the top three levels of the dendrogram
plot_dendrogram(model, names, truncate_mode="level", orientation='right')
plt.show()


# In[ ]:




