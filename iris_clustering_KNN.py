import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
from scipy import linalg
from sklearn.preprocessing import normalize
from scipy.spatial import distance
from sklearn.datasets import load_iris # load and return Iris dataset
from sklearn.manifold import TSNE
from sklearn.metrics import normalized_mutual_info_score

############Complete the code in the blank spaces#################
# Set random seed for reproducibility
#random seed
random_state= 17
# Load the Iris dataset
iris_X = load_iris().data         #data 150x4
iris_Y = load_iris().target         #Target labels
##################
#Data visualization
#Apply t-SNE to reduce the dimensionality of the iris dataset for visualization purposes

tsne = TSNE(n_components=2, init='pca', random_state=random_state)
# viasalizing high-dimensional data by giving each datapoint a location in a 2 or 3-dimensional map
# n_components (Dimension of the embedded space)
# intit : {'random', 'pca'} (Initialization the data point in the low-dimension space)
X = tsne.fit_transform(iris_X) # 150x2

# Set up a figure for visualizing different clustering results
plt.figure(figsize=(16,4))
# Plot the t-SNE output as unlabeled data
plt.subplot(171)
plt.scatter(X[:,0], X[:,1], s=10)
plt.title("Unlabeled data", fontsize=8)

# Plot the actual ground truth labels
plt.subplot(172)
iris_Y = list(iris_Y)
plt.scatter(X[:,0], X[:,1], c=iris_Y, s=10) # c=iris_Y 通过ground truth labels对每个点着色，展示ground truth labels的数据分布
plt.title("Ground Truth", fontsize=8)

y_pred_3 = KMeans(n_clusters=3, random_state=random_state).fit(X).predict(X)
y_pred_3 = list(y_pred_3)

y_pred_2 = KMeans(n_clusters=2, random_state=random_state).fit(X).predict(X)
y_pred_2 = list(y_pred_2)

y_pred_4 = KMeans(n_clusters=4, random_state=random_state).fit(X).predict(X)
y_pred_4 = list(y_pred_4)

y_pred_7 = KMeans(n_clusters=7, random_state=random_state).fit(X).predict(X)
y_pred_7 = list(y_pred_7)

y_pred_100 = KMeans(n_clusters=100, random_state=random_state).fit(X).predict(X)
y_pred_100 = list(y_pred_100)

plt.subplot(173)
plt.scatter(X[:,0], X[:,1], c=y_pred_2, s=10) # Plot KMeans clustering result
plt.title("KMeans_K=2", fontsize=8)

plt.subplot(174)
plt.scatter(X[:,0], X[:,1], c=y_pred_3, s=10) # Plot KMeans clustering result
plt.title("KMeans_K=3", fontsize=8)

plt.subplot(175)
plt.scatter(X[:,0], X[:,1], c=y_pred_4, s=10) # Plot KMeans clustering result
plt.title("KMeans_K=4", fontsize=8)

plt.subplot(176)
plt.scatter(X[:,0], X[:,1], c=y_pred_7, s=10) # Plot KMeans clustering result
plt.title("KMeans_K=7", fontsize=8)

plt.subplot(177)
plt.scatter(X[:,0], X[:,1], c=y_pred_100, s=10) # Plot KMeans clustering result
plt.title("KMeans_K=100", fontsize=8)

plt.show()

nmi_kmeans_2 = normalized_mutual_info_score(iris_Y, y_pred_2)
nmi_kmeans_3 = normalized_mutual_info_score(iris_Y, y_pred_3)
nmi_kmeans_4 = normalized_mutual_info_score(iris_Y, y_pred_4)
nmi_kmeans_7 = normalized_mutual_info_score(iris_Y, y_pred_7)
nmi_kmeans_100 = normalized_mutual_info_score(iris_Y, y_pred_100)

print('NMI (KMeans_K=2):', nmi_kmeans_2)
print('NMI (KMeans_K=3):', nmi_kmeans_3)
print('NMI (KMeans_K=4):', nmi_kmeans_4)
print('NMI (KMeans_K=7):', nmi_kmeans_7)
print('NMI (KMeans_K=100):', nmi_kmeans_100)