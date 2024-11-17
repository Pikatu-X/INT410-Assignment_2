import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import normalized_mutual_info_score


##Data Generation#######
n_samples = 1500 # Number of samples to generate

############Complete the code in the blank spaces#################
# Set random seed for reproducibility
random_state = 17
######################

# Generate isotropic Gaussian blobs for clustering
X, y = make_blobs(n_samples=n_samples, random_state=random_state) # 生成等方差（球形）高斯分布的簇
# x 生成的数据 1500x2
# y 生成的数据标签，反映属于哪个簇 1500x1

# Transformation matrix to introduce anisotropic distribution in data
transformation = [[0.60834549, -0.63667641], [-0.40887718, 0.85253229]] # 各向异性分布的变换矩阵
# Apply the transformation to the data
X_aniso = np.dot(X, transformation) # 线性变换
y = list(y)

####################################################Test_1_ElbowMethod
# 寻找最佳的簇的数量 --- 图示应为3
inertia = []
for k in range(1, 8):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_aniso)  # 假设 X 是你的数据
    inertia.append(kmeans.inertia_)
plt.plot(range(1, 8), inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()
####################################################Test_1_ElbowMethod

# Plot initial unlabeled data with anisotropy
plt.figure(figsize=(12,4))
plt.subplot(161)
plt.scatter(X_aniso[:,0], X_aniso[:,1], s=20)
plt.title("Unlabeled data", fontsize=10)

############Complete the code in the blank spaces#################
# KMeans clustering
y_pred_1 = KMeans(n_clusters=1, random_state=random_state).fit_predict(X_aniso)
y_pred_1 = list(y_pred_1)

y_pred_2 = KMeans(n_clusters=2, random_state=random_state).fit_predict(X_aniso)
y_pred_2 = list(y_pred_2)

y_pred_3 = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_aniso)
y_pred_3 = list(y_pred_3)

y_pred_4 = KMeans(n_clusters=4, random_state=random_state).fit_predict(X_aniso)
y_pred_4 = list(y_pred_4)

y_pred_10 = KMeans(n_clusters=10, random_state=random_state).fit_predict(X_aniso)
y_pred_10 = list(y_pred_10)

# Plot KMeans clustering result
plt.subplot(162)
plt.scatter(X_aniso[:,0], X_aniso[:,1], c=y_pred_1, s=20) # Plot KMeans clustering result
plt.title("KMeans_K=1", fontsize=10)

plt.subplot(163)
plt.scatter(X_aniso[:,0], X_aniso[:,1], c=y_pred_2, s=20) # Plot KMeans clustering result
plt.title("KMeans_K=2", fontsize=10)

plt.subplot(164)
plt.scatter(X_aniso[:,0], X_aniso[:,1], c=y_pred_3, s=20) # Plot KMeans clustering result
plt.title("KMeans_K=3", fontsize=10)

plt.subplot(165)
plt.scatter(X_aniso[:,0], X_aniso[:,1], c=y_pred_4, s=20) # Plot KMeans clustering result
plt.title("KMeans_K=4", fontsize=10)

plt.subplot(166)
plt.scatter(X_aniso[:,0], X_aniso[:,1], c=y_pred_10, s=20) # Plot KMeans clustering result
plt.title("KMeans_K=10", fontsize=10)
plt.show()

nmi_kmeans_1 = normalized_mutual_info_score(y, y_pred_1)
nmi_kmeans_2 = normalized_mutual_info_score(y, y_pred_2)
nmi_kmeans_3 = normalized_mutual_info_score(y, y_pred_3)
nmi_kmeans_4 = normalized_mutual_info_score(y, y_pred_4)
nmi_kmeans_10 = normalized_mutual_info_score(y, y_pred_10)


print('NMI (KMeans_K=1):', nmi_kmeans_1)
print('NMI (KMeans_K=2):', nmi_kmeans_2)
print('NMI (KMeans_K=3):', nmi_kmeans_3)
print('NMI (KMeans_K=4):', nmi_kmeans_4)
print('NMI (KMeans_K=10):', nmi_kmeans_10)