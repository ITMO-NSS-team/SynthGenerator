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


def register_individ_params(toolbox):
    toolbox.register('n_samples', random.randint,
                     SAMPLES_RANGE[0], SAMPLES_RANGE[1])
    toolbox.register('k', random.randint,
                     K_RANGE[0], K_RANGE[1])
    toolbox.register('possible_distributions', random.choice, all_distributions())
    toolbox.register('mdc_individ', tools.initCycle, creator.Individual,
                     (toolbox.n_samples, toolbox.k, toolbox.possible_distributions), n=1)


creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
register_individ_params(toolbox)
toolbox.register("population", tools.initRepeat, list, toolbox.mdc_individ)


def model_score_fitness(params, score_target=1.0):
    model_score_func = log_reg_score
    n_samples, k, distribution = params

    k = min(max(k, K_RANGE[0]), K_RANGE[1])
    n_samples = min(max(n_samples, SAMPLES_RANGE[0]), SAMPLES_RANGE[1])
    params_ = {
        'n_samples': n_samples,
        'n_feat': 2,
        'k': k,
        'possible_distributions': [distribution]
    }
    samples, labels = generated_dataset(params_)
    train_score, test_score = model_score_func(dataset=(samples, labels))

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
        'possible_distributions': [best_params[2]]
    }

    print(model_score_fitness(params=best_params))
    samples, labels = generated_dataset(params=params_)
    show_clusters(samples=samples, labels=labels)
