import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
from scipy import linalg
from sklearn.preprocessing import normalize
from scipy.spatial import distance
from iris_clustering import y_pred_GMM, spectral
from sklearn.metrics import normalized_mutual_info_score

##Data Generation#######
n_samples = 1500 # Number of samples to generate

############Complete the code in the blank spaces#################
# Set random seed for reproducibility
random_state = 17
######################################

# Generate isotropic Gaussian blobs for clustering
X, y = make_blobs(n_samples=n_samples, random_state=random_state) # 生成等方差（球形）高斯分布的簇
# x 生成的数据 1500x2
# y 生成的数据标签，反映属于哪个簇 1500x1

# Transformation matrix to introduce anisotropic distribution in data
transformation = [[0.60834549, -0.63667641], [-0.40887718, 0.85253229]] # 各向异性分布的变换矩阵
# Apply the transformation to the data
X_aniso = np.dot(X, transformation) # 线性变换
y = list(y)

# Plot initial unlabeled data with anisotropy
plt.figure(figsize=(12,4))
plt.subplot(151)
plt.scatter(X_aniso[:,0], X_aniso[:,1], s=20)
plt.title("Unlabeled data", fontsize=10)

############Complete the code in the blank spaces#################
# KMeans clustering
y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_aniso)
y_pred = list(y_pred)

# Plot KMeans clustering result
plt.subplot(152)
plt.scatter(X_aniso[:,0], X_aniso[:,1], c=y_pred, s=20) # Plot KMeans clustering result
plt.title("KMeans", fontsize=10)

# Gaussian Mixture Model clustering
y_pred_GMM = GaussianMixture(n_components=3, random_state=random_state).fit_predict(X_aniso)
y_pred_GMM = list(y_pred_GMM)

# Plot GMM clustering result
plt.subplot(153)
plt.scatter(X_aniso[:,0], X_aniso[:,1], c=y_pred_GMM, s=20) # Plot GMM clustering result
plt.title("GMM", fontsize=10)

# plt.show()

# Calculate RBF kernel matrix for spectral clustering
rbf_param = 7.6
dis_sqeuclidean = distance.cdist(X_aniso, X_aniso, metric='sqeuclidean')
K = np.exp(-dis_sqeuclidean / (2 * rbf_param ** 2))   # Calculate W_ij and form W matrix
# Calculate degree matrix for normalization
D = np.diag(K.sum(axis=1))
# Normalize the kernel matrix using the degree matrix
D_inv_sqrt = np.linalg.inv(np.sqrt(D))
M = np.dot(np.dot(D_inv_sqrt, K), D_inv_sqrt)

# Perform SVD to prepare for spectral clustering
U = linalg.svd(M)
Usubset = U[0][:,0:3] # Select the top 3 eigenvectors
# KMeans clustering on the normalized eigenvectors for spectral clustering
y_pred_sc = KMeans(n_clusters=3, random_state=random_state).fit_predict(Usubset)
y_pred_sc = list(y_pred_sc)

# Plot custom spectral clustering result
plt.subplot(154)
plt.scatter(X_aniso[:,0], X_aniso[:,1], c=y_pred_sc, s=20)# Plot custom spectral clustering result
plt.title("Spectral Clustering", fontsize=10)

# Sklearn's spectral clustering
y_pred_sc_sklearn = SpectralClustering(n_clusters=3, random_state=random_state, gamma=1 / (2 * rbf_param ** 2)).fit_predict(X_aniso)
y_pred_sc_sklearn = list(y_pred_sc_sklearn)

plt.subplot(155)
plt.scatter(X_aniso[:,0], X_aniso[:,1], c=y_pred_sc_sklearn, s=20) # Plot Sklearn's spectral clustering result
plt.title("Spectral Clustering (Sklearn)", fontsize=10)

plt.show()

#########Complete the code ############
# Define a function to calculate the Normalized Mutual Information score
# This function evaluates how similar the clustering results are to the true labels
# Print NMI
nmi_kmeans = normalized_mutual_info_score(y, y_pred)
nmi_gmm = normalized_mutual_info_score(y, y_pred_GMM)
nmi_spectral_custom = normalized_mutual_info_score(y, y_pred_sc)
nmi_spectral_sklearn = normalized_mutual_info_score(y, y_pred_sc_sklearn)

print('NMI (KMeans):', nmi_kmeans)
print('NMI (GMM):', nmi_gmm)
print('NMI (Spectral Clustering - Custom):', nmi_spectral_custom)
print('NMI (Spectral Clustering - Sklearn):', nmi_spectral_sklearn)