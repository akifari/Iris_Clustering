
import pandas as pd # open source data analysis library
import numpy as np #  used for working with arrays
import matplotlib.pyplot as plt #  plotting library for graphs
from sklearn.cluster import KMeans  # Kmeans libraires


dataset = df_iris = pd.read_csv('iris.csv') #define iris.csv as a dataset and select iris dataset
point = dataset.iloc[:, [0, 1, 2, 3 ]].values # dataset.iloc enables us to select a particular cell of the dataset. Part of pandas library. I enabled for 4 columns


from sklearn.cluster import KMeans #starting to configure k-means parameters

kmeans = KMeans(n_clusters = 3, max_iter = 99,  random_state = 0) # Create 3 claster, max itteraiton 99 like as knime examples.  
 
kmeans.fit(point) #train to my model. Compute k-means clustering.
y_kmeans = kmeans.fit_predict(point) #Compute cluster centers and predict cluster index for each sample.


dataset_outcome = pd.crosstab(index=dataset["variety"],  # Make a crosstab to define iris types on the table
                              columns="variety")      # Name the  column variety



dataset_setosa=dataset.loc[dataset["variety"]=="setosa"] #for labeled iris flower names
dataset_virginica=dataset.loc[dataset["variety"]=="virginica"]
dataset_versicolor=dataset.loc[dataset["variety"]=="versicolor"]


plt.scatter(point[y_kmeans == 0, 0], point[y_kmeans == 0, 1], s = 50, c = 'red', label = 'setosa') #plt.scstter matplotli functions for draw scatter graph
plt.scatter(point[y_kmeans == 1, 0], point[y_kmeans == 1, 1], s = 50, c = 'orange', label = 'versicolour')
plt.scatter(point[y_kmeans == 2, 0], point[y_kmeans == 2, 1], s = 50, c = 'blue', label = 'virginica')

#define centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100, c = 'black', label = 'Centroids')

plt.show() #print the graph



