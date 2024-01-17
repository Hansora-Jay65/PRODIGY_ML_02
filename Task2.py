import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Generate sample data (replace this with your actual customer purchase data)
data = {
    'CustomerID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'TotalSpent': [200, 150, 600, 300, 800, 750, 2000, 1200, 3000, 2500],
    'NumPurchases': [5, 3, 10, 6, 12, 9, 20, 15, 25, 18]
}

df = pd.DataFrame(data)

# Select features for clustering (TotalSpent and NumPurchases in this case)
X = df[['TotalSpent', 'NumPurchases']]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine the optimal number of clusters using the Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')  # Within-Cluster Sum of Squares
plt.show()

# Choose the optimal k and fit the k-means model
optimal_k = 3  # Choose based on the Elbow Method plot
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=0)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualize the clusters
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=df['Cluster'], cmap='viridis', edgecolors='k')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='X')
plt.title('Customer Segmentation based on Purchase History')
plt.xlabel('TotalSpent (Standardized)')
plt.ylabel('NumPurchases (Standardized)')
plt.show()

# Display the clustered customers
print(df[['CustomerID', 'Cluster']])
