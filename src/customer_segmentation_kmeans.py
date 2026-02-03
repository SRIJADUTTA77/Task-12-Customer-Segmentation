import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv("../dataset/mall_customers_task12.csv")

# Select features for clustering
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow Method to determine optimal k
inertia = []
K = range(1, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Save elbow plot
plt.figure()
plt.plot(K, inertia, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal k")
plt.savefig("../outputs/elbow_plot.png")
plt.close()

# Apply KMeans with optimal k = 5
kmeans = KMeans(n_clusters=5, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# Save clustering visualization
plt.figure()
plt.scatter(df["Annual Income (k$)"], df["Spending Score (1-100)"], c=df["Cluster"], cmap="viridis")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segments")
plt.savefig("../outputs/clusters_scatter.png")
plt.close()

# Save cluster summary
cluster_summary = df.groupby("Cluster").mean(numeric_only=True)
cluster_summary.to_csv("../outputs/cluster_summary.csv")

print("Task-12 Completed: Outputs saved in /outputs folder.")
