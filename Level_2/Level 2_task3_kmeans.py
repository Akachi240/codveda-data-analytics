# Codveda Internship - Level 2, Task 3: K-Means Clustering
# Dataset: Iris Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ── 1. LOAD DATASET -
df = pd.read_csv('1) iris.csv')
print("Shape:", df.shape)
print(df.head())
print("\nMissing Values:\n", df.isnull().sum())

# ── 2. PREPARE FEATURES -
X = df.drop('species', axis=1)
print("\nFeatures shape:", X.shape)

# ── 3. FEATURE SCALING -
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Scaling done!")

# ── 4. ELBOW METHOD (find optimal k) -
inertia = []
k_range = range(1, 11)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method - Optimal Number of Clusters')
plt.xticks(k_range)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('elbow_plot.png', dpi=150)
plt.show()

# ── 5. APPLY K-MEANS (k=3) -
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)
print("\nCluster counts:\n", df['Cluster'].value_counts())

# ── 6. REDUCE TO 2D WITH PCA -
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(f"\nVariance explained by 2 components: {sum(pca.explained_variance_ratio_)*100:.2f}%")

# ── 7. PLOT CLUSTERS -
colors = ['#e74c3c', '#2ecc71', '#3498db']
labels = ['Cluster 0', 'Cluster 1', 'Cluster 2']

plt.figure(figsize=(8, 6))
for i in range(3):
    mask = df['Cluster'] == i
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                c=colors[i], label=labels[i], alpha=0.7, s=80)

# Plot centroids
centroids_pca = pca.transform(kmeans.cluster_centers_)
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
            c='black', marker='X', s=200, label='Centroids')

plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('K-Means Clustering (k=3) - Iris Dataset')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('kmeans_clusters.png', dpi=150)
plt.show()

# ── 8. COMPARE CLUSTERS VS ACTUAL SPECIES -
print("\nCluster vs Actual Species:")
print(pd.crosstab(df['Cluster'], df['species']))