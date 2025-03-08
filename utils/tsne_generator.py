import numpy as np
from sklearn.manifold import TSNE
from pylab import rcParams
import matplotlib.pyplot as plt


def plot_figure(encodings, labels, save_path):
    """
    Plots a scatter plot of t-SNE results and saves the figure.

    :param encodings: t-SNE transformed encodings (2D array).
    :param labels: Labels corresponding to the encodings.
    :param save_path: Path to save the generated figure.
    """
    rcParams['figure.figsize'] = 5, 5
    plt.scatter(encodings[:, 0], encodings[:, 1], c=labels, cmap='tab10')  # Use 'tab10' colormap for better visualization
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_ticks([])
    frame1.axes.get_yaxis().set_ticks([])
    plt.tight_layout()
    plt.savefig(save_path + '_tsne.png', pad_inches=0)
    # plt.show()


def generate_tsne(encodings_path, labels_path):
    """
    Generates t-SNE embeddings for the given encodings and labels.

    :param encodings_path: Path to the encodings file (.npy).
    :param labels_path: Path to the labels file (.npy).
    """
    encodings = np.load(encodings_path, allow_pickle=True)  # Use allow_pickle=True for compatibility
    labels = np.load(labels_path, allow_pickle=True)

    tsne_model = TSNE(verbose=4)

    # Only draw a subset as we don't want to crowd the image
    subset_size = min(len(encodings), 15000)
    encodings_subset = encodings[:subset_size]
    labels_subset = labels[:subset_size]

    tsne_results = tsne_model.fit_transform(encodings_subset)
    plot_figure(tsne_results, labels_subset, encodings_path)


if __name__ == '__main__':
    generate_tsne('../data/mnist_train_encodings.npy', '../data/mnist_train_encodings_labels.npy')
    # generate_tsne('../data/fsdd_train_encodings.npy', '../data/fsdd_train_encodings_labels.npy')