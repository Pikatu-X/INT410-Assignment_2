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

# rbf_para = 1, K = 3
spectral_1 = SpectralClustering(n_clusters=3, gamma=1/(2 * 1 ** 2),random_state=random_state)
y_pred_sc_sklearn_1 = spectral_1.fit_predict(X)
y_pred_sc_sklearn_1 = list(y_pred_sc_sklearn_1)

# rbf_para = 2, K = 3
spectral_2 = SpectralClustering(n_clusters=3, gamma=1/(2 * 2 ** 2),random_state=random_state)
y_pred_sc_sklearn_2 = spectral_2.fit_predict(X)
y_pred_sc_sklearn_2 = list(y_pred_sc_sklearn_2)

# rbf_para = 3, K = 3
spectral_3 = SpectralClustering(n_clusters=3, gamma=1/(2 * 3 ** 2),random_state=random_state)
y_pred_sc_sklearn_3 = spectral_3.fit_predict(X)
y_pred_sc_sklearn_3 = list(y_pred_sc_sklearn_3)

# rbf_para = 100, K = 3
spectral_100 = SpectralClustering(n_clusters=3, gamma=1/(2 * 1000 ** 2),random_state=random_state)
y_pred_sc_sklearn_100 = spectral_100.fit_predict(X)
y_pred_sc_sklearn_100 = list(y_pred_sc_sklearn_100)

#
plt.subplot(163)
plt.scatter(X[:,0], X[:,1], c=y_pred_sc_sklearn_1, s=10) # c=iris_Y 通过ground truth labels对每个点着色，展示ground truth labels的数据分布
plt.title("rbf_para=1, K=3", fontsize=8)

plt.subplot(164)
plt.scatter(X[:,0], X[:,1], c=y_pred_sc_sklearn_2, s=10) # c=iris_Y 通过ground truth labels对每个点着色，展示ground truth labels的数据分布
plt.title("rbf_para=2, K=3", fontsize=8)

plt.subplot(165)
plt.scatter(X[:,0], X[:,1], c=y_pred_sc_sklearn_3, s=10) # c=iris_Y 通过ground truth labels对每个点着色，展示ground truth labels的数据分布
plt.title("rbf_para=3, K=3", fontsize=8)

plt.subplot(166)
plt.scatter(X[:,0], X[:,1], c=y_pred_sc_sklearn_100, s=10) # c=iris_Y 通过ground truth labels对每个点着色，展示ground truth labels的数据分布
plt.title("rbf_para=100, K=3", fontsize=8)

plt.show()

nmi_spectral_1 = normalized_mutual_info_score(iris_Y, y_pred_sc_sklearn_1)
nmi_spectral_2 = normalized_mutual_info_score(iris_Y, y_pred_sc_sklearn_2)
nmi_spectral_3 = normalized_mutual_info_score(iris_Y, y_pred_sc_sklearn_3)
nmi_spectral_100 = normalized_mutual_info_score(iris_Y, y_pred_sc_sklearn_100)


print('NMI (rbf_para=1, K=3):', nmi_spectral_1)
print('NMI (rbf_para=2, K=3):', nmi_spectral_2)
print('NMI (rbf_para=3, K=3):', nmi_spectral_3)
print('NMI (rbf_para=100, K=3):', nmi_spectral_100)
