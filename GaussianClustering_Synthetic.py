import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
from scipy import linalg
from sklearn.preprocessing import normalize
from scipy.spatial import distance
# from iris_clustering import y_pred_GMM, spectral
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

y_pred_GMM_KMeans_3 = GaussianMixture(n_components=3, random_state=random_state).fit_predict(X_aniso)
y_pred_GMM_KMeans_3 = list(y_pred_GMM_KMeans_3)

y_pred_GMM_KMeans_2 = GaussianMixture(n_components=2, random_state=random_state).fit_predict(X_aniso)
y_pred_GMM_KMeans_2 = list(y_pred_GMM_KMeans_2)

y_pred_GMM_KMeans_5 = GaussianMixture(n_components=5, random_state=random_state).fit_predict(X_aniso)
y_pred_GMM_KMeans_5 = list(y_pred_GMM_KMeans_5)

y_pred_GMM_KMeans_10 = GaussianMixture(n_components=10, random_state=random_state).fit_predict(X_aniso)
y_pred_GMM_KMeans_10 = list(y_pred_GMM_KMeans_10)

# plt.subplot(152)
# plt.scatter(X_aniso[:,0], X_aniso[:,1], c=y_pred_GMM_KMeans_2, s=20) # Plot KMeans clustering result
# plt.title("GMM_KMeans_K=2", fontsize=10)
#
# plt.subplot(153)
# plt.scatter(X_aniso[:,0], X_aniso[:,1], c=y_pred_GMM_KMeans_3, s=20) # Plot KMeans clustering result
# plt.title("GMM_KMeans_K=3", fontsize=10)
#
# plt.subplot(154)
# plt.scatter(X_aniso[:,0], X_aniso[:,1], c=y_pred_GMM_KMeans_5, s=20) # Plot KMeans clustering result
# plt.title("GMM_KMeans_K=5", fontsize=10)
#
# plt.subplot(155)
# plt.scatter(X_aniso[:,0], X_aniso[:,1], c=y_pred_GMM_KMeans_10, s=20) # Plot KMeans clustering result
# plt.title("GMM_KMeans_K=10", fontsize=10)



y_pred_GMM_random_3 = GaussianMixture(n_components=3, init_params='random', random_state=random_state).fit(X).predict(X)
y_pred_GMM_random_3 = list(y_pred_GMM_random_3)

y_pred_GMM_random_2 = GaussianMixture(n_components=2, init_params='random', random_state=random_state).fit(X).predict(X)
y_pred_GMM_random_2 = list(y_pred_GMM_random_2)

y_pred_GMM_random_5 = GaussianMixture(n_components=5, init_params='random', random_state=random_state).fit(X).predict(X)
y_pred_GMM_random_5 = list(y_pred_GMM_random_5)

y_pred_GMM_random_10 = GaussianMixture(n_components=10, init_params='random', random_state=random_state).fit(X).predict(X)
y_pred_GMM_random_10 = list(y_pred_GMM_random_10)

plt.subplot(152)
plt.scatter(X_aniso[:,0], X_aniso[:,1], c=y_pred_GMM_random_2, s=20) # Plot KMeans clustering result
plt.title("GMM_Random_K=2", fontsize=10)

plt.subplot(153)
plt.scatter(X_aniso[:,0], X_aniso[:,1], c=y_pred_GMM_random_3, s=20) # Plot KMeans clustering result
plt.title("GMM_Random_K=3", fontsize=10)

plt.subplot(154)
plt.scatter(X_aniso[:,0], X_aniso[:,1], c=y_pred_GMM_random_5, s=20) # Plot KMeans clustering result
plt.title("GMM_Random_K=5", fontsize=10)

plt.subplot(155)
plt.scatter(X_aniso[:,0], X_aniso[:,1], c=y_pred_GMM_random_10, s=20) # Plot KMeans clustering result
plt.title("GMM_Random_K=10", fontsize=10)

plt.show()

nmi_gmm_3 = normalized_mutual_info_score(y, y_pred_GMM_KMeans_3)
nmi_gmm_2 = normalized_mutual_info_score(y, y_pred_GMM_KMeans_2)
nmi_gmm_5 = normalized_mutual_info_score(y, y_pred_GMM_KMeans_5)
nmi_gmm_10 = normalized_mutual_info_score(y, y_pred_GMM_KMeans_10)

nmi_gmm_random_3 = normalized_mutual_info_score(y, y_pred_GMM_random_3)
nmi_gmm_random_2 = normalized_mutual_info_score(y, y_pred_GMM_random_2)
nmi_gmm_random_5 = normalized_mutual_info_score(y, y_pred_GMM_random_5)
nmi_gmm_random_10 = normalized_mutual_info_score(y, y_pred_GMM_random_10)

print('NMI (GMM KMeans_k=3):', nmi_gmm_3)
print('NMI (GMM KMeans_k=2):', nmi_gmm_2)
print('NMI (GMM KMeans_k=5):', nmi_gmm_5)
print('NMI (GMM KMeans_k=10):', nmi_gmm_10)

print('NMI (GMM Random):', nmi_gmm_random_3)
print('NMI (GMM Random_2):', nmi_gmm_random_2)
print('NMI (GMM Random_5):', nmi_gmm_random_5)
print('NMI (GMM Random_10):', nmi_gmm_random_10)