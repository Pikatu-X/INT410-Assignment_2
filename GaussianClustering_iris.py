import os
os.environ['OMP_NUM_THREADS'] = '1'

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
plt.subplot(161)
plt.scatter(X[:,0], X[:,1], s=10)
plt.title("Unlabeled data", fontsize=8)

# Plot the actual ground truth labels
plt.subplot(162)
iris_Y = list(iris_Y)
plt.scatter(X[:,0], X[:,1], c=iris_Y, s=10) # c=iris_Y 通过ground truth labels对每个点着色，展示ground truth labels的数据分布
plt.title("Ground Truth", fontsize=8)


y_pred_GMM = GaussianMixture(n_components=3, random_state=random_state).fit(X).predict(X)   # Apply Gaussian Mixture Model clustering with 3 components
y_pred_GMM = list(y_pred_GMM)

y_pred_GMM_2 = GaussianMixture(n_components=2, random_state=random_state).fit(X).predict(X)   # Apply Gaussian Mixture Model clustering with 3 components
y_pred_GMM_2 = list(y_pred_GMM_2)

y_pred_GMM_5 = GaussianMixture(n_components=5, random_state=random_state).fit(X).predict(X)   # Apply Gaussian Mixture Model clustering with 3 components
y_pred_GMM_5 = list(y_pred_GMM_5)

y_pred_GMM_10 = GaussianMixture(n_components=10, random_state=random_state).fit(X).predict(X)   # Apply Gaussian Mixture Model clustering with 3 components
y_pred_GMM_10 = list(y_pred_GMM_10)


y_pred_GMM_random = GaussianMixture(n_components=3, init_params='random', random_state=random_state).fit(X).predict(X)
y_pred_GMM_random = list(y_pred_GMM_random)

y_pred_GMM_random_2 = GaussianMixture(n_components=2, init_params='random', random_state=random_state).fit(X).predict(X)
y_pred_GMM_random_2 = list(y_pred_GMM_random_2)

y_pred_GMM_random_5 = GaussianMixture(n_components=5, init_params='random', random_state=random_state).fit(X).predict(X)
y_pred_GMM_random_5 = list(y_pred_GMM_random_5)

y_pred_GMM_random_10 = GaussianMixture(n_components=10, init_params='random', random_state=random_state).fit(X).predict(X)
y_pred_GMM_random_10 = list(y_pred_GMM_random_10)

# plt.subplot(163)
# plt.scatter(X[:,0], X[:,1], c=y_pred_GMM_2, s=10)# Plot GMM clustering result
# plt.title("GMM_KMeans_K=2", fontsize=8)
#
# plt.subplot(164)
# plt.scatter(X[:,0], X[:,1], c=y_pred_GMM, s=10)# Plot GMM clustering result
# plt.title("GMM_KMeans_K=3", fontsize=8)
#
# plt.subplot(165)
# plt.scatter(X[:,0], X[:,1], c=y_pred_GMM_5, s=10)# Plot GMM clustering result
# plt.title("GMM_KMeans_K=5", fontsize=8)
#
# plt.subplot(166)
# plt.scatter(X[:,0], X[:,1], c=y_pred_GMM_random_10, s=10)# Plot GMM clustering result
# plt.title("GMM_KMeans_K=10", fontsize=8)
# plt.show()

plt.subplot(163)
plt.scatter(X[:,0], X[:,1], c=y_pred_GMM_random_2, s=10)# Plot GMM clustering result
plt.title("GMM_Random_K=2", fontsize=8)

plt.subplot(164)
plt.scatter(X[:,0], X[:,1], c=y_pred_GMM_random, s=10)# Plot GMM clustering result
plt.title("GMM_Random_K=3", fontsize=8)

plt.subplot(165)
plt.scatter(X[:,0], X[:,1], c=y_pred_GMM_random_5, s=10)# Plot GMM clustering result
plt.title("GMM_Random_K=5", fontsize=8)

plt.subplot(166)
plt.scatter(X[:,0], X[:,1], c=y_pred_GMM_random_10, s=10)# Plot GMM clustering result
plt.title("GMM_Random_K=10", fontsize=8)
plt.show()


nmi_gmm = normalized_mutual_info_score(iris_Y, y_pred_GMM)
nmi_gmm_2 = normalized_mutual_info_score(iris_Y, y_pred_GMM_2)
nmi_gmm_5 = normalized_mutual_info_score(iris_Y, y_pred_GMM_5)
nmi_gmm_10 = normalized_mutual_info_score(iris_Y, y_pred_GMM_10)

nmi_gmm_random = normalized_mutual_info_score(iris_Y, y_pred_GMM_random)
nmi_gmm_random_2 = normalized_mutual_info_score(iris_Y, y_pred_GMM_random_2)
nmi_gmm_random_5 = normalized_mutual_info_score(iris_Y, y_pred_GMM_random_5)
nmi_gmm_random_10 = normalized_mutual_info_score(iris_Y, y_pred_GMM_random_10)

print('NMI (GMM KMeans_k=3):', nmi_gmm)
print('NMI (GMM KMeans_k=2):', nmi_gmm_2)
print('NMI (GMM KMeans_k=5):', nmi_gmm_5)
print('NMI (GMM KMeans_k=10):', nmi_gmm_10)

print('NMI (GMM Random):', nmi_gmm_random)
print('NMI (GMM Random_2):', nmi_gmm_random_2)
print('NMI (GMM Random_5):', nmi_gmm_random_5)
print('NMI (GMM Random_10):', nmi_gmm_random_10)