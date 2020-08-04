from mdcgenpy.clusters import ClusterGenerator

default_params = {
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


def generated_dataset(params=None):
    if params is None:
        params = default_params

    cluster_gen = ClusterGenerator(**params)
    samples, labels = cluster_gen.generate_data()

    return samples, labels
