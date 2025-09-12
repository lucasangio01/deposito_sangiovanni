import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(path):
    data = pd.read_csv(path)
    return data

def standardize(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def apply_kmeans(X_scaled):
    silhouette_scores = []
    K = range(2, 11)

    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        silhouette_scores.append(score)
    
    return K, silhouette_scores, kmeans, labels

def plot_silhouette(K, silhouette_scores):
    plt.figure(figsize=(8, 5))
    sns.lineplot(x=list(K), y=silhouette_scores, marker='o')
    plt.title("Silhouette Score al variare di k")
    plt.xlabel("Numero di cluster (k)")
    plt.ylabel("Silhouette Score")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_datapoints_clusters(X_scaled, kmeans, labels):
    plt.figure(figsize=(6, 6))
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                c='red', marker='X', s=200, label='Centroidi')
    plt.title("Cluster trovati con k-Means")
    plt.xlabel("Annual income")
    plt.ylabel("Spending")
    plt.legend()
    plt.grid(True)
    plt.show()

def new_kmeans(X_scaled):
    kmeans_5 = KMeans(n_clusters=5, random_state=42)
    labels_5 = kmeans_5.fit_predict(X_scaled)
    return kmeans_5, labels_5