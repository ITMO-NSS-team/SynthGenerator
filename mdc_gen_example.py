import matplotlib.pyplot as plt
from mdcgenpy.clusters import ClusterGenerator
from mdcgenpy.clusters.distributions import valid_distributions


def show_clusters(samples, labels):
    plt.scatter(samples[:, 0], samples[:, 1], c=labels, s=10)
    plt.show()


def all_distributions():
    return list(valid_distributions)


if __name__ == '__main__':
    print(all_distributions())
    params = {
        'n_samples': 2000,
        'n_feats': 2,
        'k': 3,
        'min_samples': 0,
        'possible_distributions': ['gaussian', 'gamma'],
        'corr': 0.,
        'compactness_factor': 0.1,
        'alpha_n': 1,
        'outliers': 50,
        'ki_coeff3': 3.
    }
    cluster_gen = ClusterGenerator(**params)

    # Get tuple with a numpy array with samples and another with labels
    samples, labels = cluster_gen.generate_data()
    show_clusters(samples=samples, labels=labels)
