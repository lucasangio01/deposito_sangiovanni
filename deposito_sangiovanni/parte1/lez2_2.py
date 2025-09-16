import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


def load_data(path):
    data = pd.read_csv(path).drop(["Channel", "Region"], axis = 1)
    return data


def scale_data(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    return data_scaled


def choose_min_samples(data_scaled):
    silhouette_scores = []
    M = range(3, 8)

    for m in M:
        dbscan_labels = make_dbscan(data_scaled, min_samples = m)
        score = silhouette_score(data_scaled, dbscan_labels)
        silhouette_scores.append(score)

    plt.figure(figsize=(5, 4))
    sns.lineplot(x=list(M), y=silhouette_scores, marker='o')
    plt.title("Silhouette Score al variare di min_samples")
    plt.xlabel("Numero di min_samples")
    plt.ylabel("Silhouette Score")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def apply_pca(data_scaled):
    pca = PCA(n_components=2, random_state=42)
    data_scaled_pca = pca.fit_transform(data_scaled)
    return data_scaled_pca


def make_dbscan(data_scaled, min_samples):
    dbscan = DBSCAN(eps=0.8, min_samples=min_samples)
    dbscan_labels = dbscan.fit_predict(data_scaled)
    return dbscan_labels


def plot_dbscan(data_scaled_pca, dbscan_labels):
    unique_labels = set(dbscan_labels)
    palette = sns.color_palette("Set2", len(unique_labels))
    color_map = {
        label: palette[i] if label != -1 else (0.6, 0.6, 0.6)  # grigio per outlier
        for i, label in enumerate(sorted(unique_labels))
    }
    colors_dbscan = [color_map[label] for label in dbscan_labels]
    
    plt.figure(figsize=(5, 3))
    plt.scatter(data_scaled_pca[:, 0], data_scaled_pca[:, 1], c=colors_dbscan, s=40, edgecolor='black')
    plt.title("DBSCAN con PCA: forma naturale + outlier (grigio)")
    plt.xlabel("Componente 1")
    plt.ylabel("Componente 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def compute_silhouette_avg(data_scaled):
    silhouette_avg = silhouette_score(data_scaled, labels = make_dbscan(data_scaled, min_samples = 5))
    return silhouette_avg
