import time

import numpy as np
import matplotlib.pyplot as plt
from IPython.utils.timing import clock
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
from scipy import linalg
from sklearn.preprocessing import normalize
from scipy.spatial import distance
from sklearn.metrics import normalized_mutual_info_score
# from iris_clustering import y_pred_GMM, spectral

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




startTime_2 = time.time()
Usubset_2 = U[0][:,0:2] # Select the top 3 eigenvectors
# KMeans clustering on the normalized eigenvectors for spectral clustering
startTime_2 = time.time()
y_pred_sc_2 = KMeans(n_clusters=3, random_state=random_state).fit_predict(Usubset_2)
endTime_2 = time.time()
y_pred_sc_2 = list(y_pred_sc_2)



Usubset_1 = U[0][:,0:1] # Select the top 3 eigenvectors
# KMeans clustering on the normalized eigenvectors for spectral clustering
startTime_1 = time.time()
y_pred_sc_1 = KMeans(n_clusters=3, random_state=random_state).fit_predict(Usubset_1)
endTime_1 = time.time()
y_pred_sc_1 = list(y_pred_sc_1)

Usubset_3 = U[0][:,0:3] # Select the top 3 eigenvectors
# KMeans clustering on the normalized eigenvectors for spectral clustering
startTime_3 = time.time()
y_pred_sc_3 = KMeans(n_clusters=3, random_state=random_state).fit_predict(Usubset_3)
endTime_3 = time.time()
y_pred_sc_3 = list(y_pred_sc_3)

plt.subplot(152)
plt.scatter(X_aniso[:,0], X_aniso[:,1], c=y_pred_sc_1, s=20)# Plot custom spectral clustering result
plt.title("Select 1 Eignvectors", fontsize=10)

plt.subplot(153)
plt.scatter(X_aniso[:,0], X_aniso[:,1], c=y_pred_sc_2, s=20)# Plot custom spectral clustering result
plt.title("Select 2 Eignvectors", fontsize=10)

plt.subplot(154)
plt.scatter(X_aniso[:,0], X_aniso[:,1], c=y_pred_sc_3, s=20)# Plot custom spectral clustering result
plt.title("Select 3 Eignvectors", fontsize=10)

plt.show()

nmi_spectral_1 = normalized_mutual_info_score(y, y_pred_sc_1)
nmi_spectral_2 = normalized_mutual_info_score(y, y_pred_sc_2)
nmi_spectral_3 = normalized_mutual_info_score(y, y_pred_sc_3)

print('NMI (1 vector):', nmi_spectral_1)
print('NMI (2 vector):', nmi_spectral_2)
print('NMI (3 vector):', nmi_spectral_3)

elaspsed_time_1 = endTime_1 - startTime_1
elaspsed_time_2 = endTime_2 - startTime_2
elaspsed_time_3 = endTime_3 - startTime_3
print(f"特征向量数量为1，运行时间：{elaspsed_time_1:.4f} 秒")
print(f"特征向量数量为2，运行时间：{elaspsed_time_2:.4f} 秒")
print(f"特征向量数量为3，运行时间：{elaspsed_time_3:.4f} 秒")