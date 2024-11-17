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
plt.subplot(161)
plt.scatter(X[:,0], X[:,1], s=10)
plt.title("Unlabeled data", fontsize=8)

# Plot the actual ground truth labels
plt.subplot(162)
iris_Y = list(iris_Y)
plt.scatter(X[:,0], X[:,1], c=iris_Y, s=10) # c=iris_Y 通过ground truth labels对每个点着色，展示ground truth labels的数据分布
plt.title("Ground Truth", fontsize=8)

# plt.show() # 测试

#########Complete the code in the blank spaces############
# kmeans = KMeans(n_clusters=3, random_state=random_state) # Apply KMeans clustering with 3 clusters
# y_pred = kmeans.fit_predict(iris_X)
y_pred = KMeans(n_clusters=3, random_state=random_state).fit(X).predict(X)
y_pred = list(y_pred)

# Plot KMeans clustering result
plt.subplot(163)
plt.scatter(X[:,0], X[:,1], c=y_pred, s=10) # Plot KMeans clustering result
plt.title("KMeans", fontsize=8)
#####################################
# plt.show() # 测试

#################################### test 1
# 测试一下区别 fit(X).Predict(X)和 fit_predict(iris_X)
# kmeans = KMeans(n_clusters=3, random_state=random_state) # Apply KMeans clustering with 3 clusters
# y_pred = kmeans.fit_predict(X)
# y_pred = list(y_pred)
#
# plt.subplot(164)
# plt.scatter(X[:,0], X[:,1], c=y_pred, s=10) # Plot KMeans clustering result
# plt.title("KMeans-fit_predict", fontsize=8)
#
# plt.show()
#################################### test 1

#########Complete the code in the blank spaces############
y_pred_GMM = GaussianMixture(n_components=3, random_state=random_state).fit(X).predict(X)   # Apply Gaussian Mixture Model clustering with 3 components
y_pred_GMM = list(y_pred_GMM)

# Plot GMM clustering result
plt.subplot(164)
plt.scatter(X[:,0], X[:,1], c=y_pred_GMM, s=10)# Plot GMM clustering result
plt.title("GMM", fontsize=8)

# plt.show() 测试

#Calculate RBF kernel matrix
rbf_param = 2
dis_sqeuclidean = pdist(X, metric='sqeuclidean')
dis_sqeuclidean = squareform(dis_sqeuclidean)
K = np.exp(-dis_sqeuclidean / (2 * rbf_param ** 2))  # Calculate W_ij and form W matrix
# Compute the degree matrix D
D = np.diag(np.sum(K, axis=1))
# Normalize the kernel matrix
D_inv_sqrt = np.linalg.inv(np.sqrt(D))
M = np.dot(np.dot(D_inv_sqrt, K), D_inv_sqrt)

# Perform SVD to get spectral clustering input
U = linalg.svd(M)
Usubset = U[0][:,0:3]
# Apply KMeans clustering on the normalized eigenvectors
y_pred_sc = KMeans(n_clusters=3, random_state=random_state).fit_predict(Usubset)
y_pred_sc = list(y_pred_sc)


plt.subplot(165)
plt.scatter(X[:,0], X[:,1], c=y_pred_sc, s=10) # Plot Spectral Clustering result using custom kernel-based method
plt.title("Spectral Clustering", fontsize=8)

# Apply Spectral Clustering using Sklearn
spectral = SpectralClustering(n_clusters=3, gamma=1/(2 * rbf_param ** 2),random_state=random_state)
y_pred_sc_sklearn = spectral.fit_predict(X)
y_pred_sc_sklearn = list(y_pred_sc_sklearn)

plt.subplot(166)
plt.scatter(X[:,0], X[:,1], c=y_pred_sc, s=10)# Plot Spectral Clustering result
plt.title("Spectral Clustering (Sklearn)", fontsize=8)

plt.show()

#########Complete the code ############
# Define a function to calculate the Normalized Mutual Information score
# This function evaluates how similar the clustering results are to the true labels
#Print NMI
nmi_kmeans = normalized_mutual_info_score(iris_Y, y_pred)
nmi_gmm = normalized_mutual_info_score(iris_Y, y_pred_GMM)
nmi_spectral_custom = normalized_mutual_info_score(iris_Y, y_pred_sc)
nmi_spectral_sklearn = normalized_mutual_info_score(iris_Y, y_pred_sc_sklearn)

print('NMI (KMeans):', nmi_kmeans)
print('NMI (GMM):', nmi_gmm)
print('NMI (Spectral Clustering - Custom):', nmi_spectral_custom)
print('NMI (Spectral Clustering - Sklearn):', nmi_spectral_sklearn)
