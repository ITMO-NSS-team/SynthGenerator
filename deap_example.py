import random

import numpy as np
from deap import algorithms
from deap import base
from deap import creator
from deap import tools

from fit_models import log_reg_score
from generators.mdc import generated_dataset
from mdc_gen_example import all_distributions, show_clusters

SAMPLES_RANGE = [1000, 2000]
K_RANGE = [1, 10]
ALPHA = list(range(1, 5))
OUTLIERS_RANGE = [50, 1000]


def register_individ_params(toolbox):
    toolbox.register('n_samples', random.randint,
                     SAMPLES_RANGE[0], SAMPLES_RANGE[1])
    toolbox.register('k', random.randint,
                     K_RANGE[0], K_RANGE[1])
    toolbox.register('possible_distributions', random.choice, all_distributions())
    toolbox.register('corr', random.random)
    toolbox.register('compactness_factor', random.random)
    toolbox.register('alpha_n', random.choice, ALPHA)
    toolbox.register('outliers', random.randint,
                     OUTLIERS_RANGE[0], OUTLIERS_RANGE[1])

    toolbox.register('mdc_individ', tools.initCycle, creator.Individual,
                     (toolbox.n_samples, toolbox.k,
                      toolbox.possible_distributions,
                      toolbox.corr, toolbox.compactness_factor,
                      toolbox.alpha_n, toolbox.outliers), n=1)


creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
register_individ_params(toolbox)
toolbox.register("population", tools.initRepeat, list, toolbox.mdc_individ)


def model_score_fitness(params, score_target=1.0):
    model_score_func = log_reg_score
    n_samples, k, distribution, corr, compact, alpha_n, outliers = params

    k = min(max(k, K_RANGE[0]), K_RANGE[1])
    n_samples = min(max(n_samples, SAMPLES_RANGE[0]), SAMPLES_RANGE[1])
    corr = min(max(corr, 0.0), 1.0)
    alpha_n = int(alpha_n)
    alpha_n = min(max(alpha_n, ALPHA[0]), ALPHA[1])
    outliers = int(outliers)
    outliers = min(max(outliers, OUTLIERS_RANGE[0]), OUTLIERS_RANGE[1])

    params_ = {
        'n_samples': n_samples,
        'n_feat': 2,
        'k': k,
        'possible_distributions': [distribution],
        'corr': corr,
        'compactness_factor': compact,
        'alpha_n': alpha_n,
        'outliers': outliers
    }
    samples, labels = generated_dataset(params_)

    # cv_samples, cv_labels, valid_samples, valid_labels = train_test_split(samples, labels, test_size=0.2)
    train_score, _ = model_score_func(dataset=(samples, labels))

    fitness = np.abs(score_target - train_score)
    return fitness


def eval_fitness(individual):
    score = model_score_fitness(individual)
    return score,


if __name__ == '__main__':
    toolbox.register("evaluate", eval_fitness)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=20)

    NGEN = 30
    for gen in range(NGEN):
        print(f'gen # {gen}')
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = toolbox.select(offspring, k=len(population))
    top10 = tools.selBest(population, k=10)

    print(top10)

    best_params = top10[0]

    params_ = {
        'n_samples': best_params[0],
        'n_feat': 2,
        'k': best_params[1],
        'possible_distributions': [best_params[2]],
        'corr': best_params[3],
        'compactness_factor': best_params[4],
        'alpha_n': best_params[5],
        'outliers': best_params[6]
    }

    print(model_score_fitness(params=best_params))
    samples, labels = generated_dataset(params=params_)
    show_clusters(samples=samples, labels=labels)
