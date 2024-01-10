import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def visualize_tsne(features, labels, save_path):
    # Perform t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    reduced_features = tsne.fit_transform(features)

    # Plot the 2D t-SNE visualization
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap="viridis")
    plt.title("t-SNE Visualization of Intermediate Features")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.colorbar(label="Class Labels")
    plt.savefig(save_path)
    plt.close()
