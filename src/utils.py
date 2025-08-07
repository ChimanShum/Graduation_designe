import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
import numpy as np
from sklearn.preprocessing import StandardScaler

def plot_kde(data, color, label):
    kde = gaussian_kde(data)
    x = np.linspace(min(data), max(data), 1000)
    plt.plot(x, kde(x), color=color, label=label, linewidth=2)

def plot_pca(X_original, X_synthetic, title):
    scaler = StandardScaler()
    X_original_scaled = scaler.fit_transform(X_original)
    X_synthetic_scaled = scaler.transform(X_synthetic)

    pca = PCA(n_components=2)
    X_original_pca = pca.fit_transform(X_original_scaled)
    X_synthetic_pca = pca.transform(X_synthetic_scaled)

    plt.figure(figsize=(10, 6))
    plt.scatter(X_original_pca[:, 0], X_original_pca[:, 1], alpha=0.5, label="原始数据", color='blue')
    plt.scatter(X_synthetic_pca[:, 0], X_synthetic_pca[:, 1], alpha=0.5, label="生成数据", color='orange')
    plt.title(f"PCA：{title}")
    plt.xlabel('主成分1')
    plt.ylabel('主成分2')
    plt.legend()
    plt.show()
