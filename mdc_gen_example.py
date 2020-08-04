import matplotlib.pyplot as plt
from mdcgenpy.clusters import ClusterGenerator


def show_clusters(samples, labels):
    plt.scatter(samples[:, 0], samples[:, 1], c=labels, s=10)
    plt.show()


if __name__ == '__main__':
    cluster_gen = ClusterGenerator(n_samples=2000, n_feats=2, k=3)

    # Get tuple with a numpy array with samples and another with labels
    samples, labels = cluster_gen.generate_data()
    show_clusters(samples=samples, labels=labels)
