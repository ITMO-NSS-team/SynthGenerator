import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution

from fit_models import log_reg_score
from generators.mdc import generated_dataset


def show_clusters(samples, labels):
    plt.scatter(samples[:, 0], samples[:, 1], c=labels, s=10)
    plt.show()


def model_score_fitness(params, score_target=1.0):
    model_score_func = log_reg_score
    k, outliers, compactness = params
    params_ = {
        'n_samples': 5000,
        'n_feat': 2,
        'k': int(k),
        'outliers': int(outliers),
        'compactness_factor': compactness

    }
    samples, labels = generated_dataset(params_)
    train_score, test_score = model_score_func(dataset=(samples, labels))

    fitness = np.abs(train_score - score_target)
    print(fitness)

    return train_score


bounds = [(2, 10), (50, 500), (0.05, 1.0)]
result = differential_evolution(model_score_fitness, bounds,
                                strategy='rand2exp',
                                tol=0.05, maxiter=100)

print(result.x)
print(result.fun)

k, outliers, compactness = result.x
params_ = {
    'n_samples': 10000,
    'n_feat': 10,
    'k': int(k),
    'outliers': int(outliers),
    'compactness_factor': compactness

}
samples, labels = generated_dataset(params_)
show_clusters(samples, labels)
