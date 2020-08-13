import random

import matplotlib.pyplot as plt
import numpy as np
from deap import algorithms
from deap import base
from deap import creator
from deap import tools

from fit_models import log_reg_score, dec_tree_score
from generators.mdc import generated_dataset
from mdc_gen_example import all_distributions, show_clusters

SAMPLES_RANGE = [1000, 2000]
K_RANGE = [1, 10]
ALPHA = list(range(1, 5))
OUTLIERS_RANGE = [50, 1000]


def mdc_gen_params():
    params = ['n_samples', 'k', 'possible_distributions',
              'corr', 'compactness_factor', 'alpha_n',
              'outliers']

    return params


def individ_to_params(individ):
    extracted_params = {}
    for gen, param_name in zip(individ, mdc_gen_params()):
        if param_name is 'possible_distributions':
            extracted_params[param_name] = [gen]
        else:
            extracted_params[param_name] = gen

    return extracted_params


def fix_out_of_ranges(params):
    params['n_samples'] = min(max(params['n_samples'], SAMPLES_RANGE[0]), SAMPLES_RANGE[1])
    params['k'] = min(max(params['k'], K_RANGE[0]), K_RANGE[1])
    params['corr'] = min(max(params['corr'], 0.0), 1.0)
    params['alpha_n'] = int(min(max(params['alpha_n'], ALPHA[0]), ALPHA[1]))
    params['outliers'] = int(min(max(params['outliers'], OUTLIERS_RANGE[0]), OUTLIERS_RANGE[1]))


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


creator.create("FitnessMin", base.Fitness, weights=(-1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()
register_individ_params(toolbox)
toolbox.register("population", tools.initRepeat, list, toolbox.mdc_individ)


def model_score_fitness(params, score_target=1.0):
    model_score_func = log_reg_score

    params_ = individ_to_params(params)
    fix_out_of_ranges(params_)
    params_['n_feat'] = 2
    samples, labels = generated_dataset(params_)

    train_score, _ = model_score_func(dataset=(samples, labels))

    fitness = np.abs(score_target - train_score)
    return fitness


def two_model_score_fitness(params,
                            score_target_first=1.0, score_target_second=1.0):
    first_model_score = log_reg_score
    second_model_score = dec_tree_score

    params_ = individ_to_params(params)
    fix_out_of_ranges(params_)
    params_['n_feat'] = 2
    samples, labels = generated_dataset(params_)

    first_score, _ = first_model_score(dataset=(samples, labels))
    second_score, _ = second_model_score(dataset=(samples, labels))

    print(f'{first_score}, {second_score}')
    # fitness_first = np.abs(score_target_first - first_score)
    # fitness_second = np.abs(score_target_second - second_score)

    return first_score, second_score


def eval_fitness(individual):
    score = model_score_fitness(individual)
    return score,


def eval_multi_fitness(individual):
    score = two_model_score_fitness(individual)

    return score


def run_evolution(generations=10):
    toolbox.register("evaluate", eval_multi_fitness)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=20)

    fit_history_first = []
    fit_history_second = []
    for gen in range(generations):
        print(f'gen # {gen}')
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = toolbox.select(offspring, k=len(population))

        fit_values_first = []
        fit_values_second = []
        for ind in population:
            first_, second_ = ind.fitness.values
            fit_values_first.append(first_)
            fit_values_second.append(second_)

        plt.scatter(fit_values_first, fit_values_second)
        plt.show()
        fit_history_first.append(np.average(fit_values_first))
        fit_history_second.append(np.average(fit_values_second))

    top10 = tools.selBest(population, k=10)

    return top10, (fit_history_first, fit_history_second)


def show_fitness_history(history):
    gens = [gen for gen in range(len(history))]

    plt.plot(gens, history)
    plt.show()


if __name__ == '__main__':
    top10, history = run_evolution(generations=20)
    print(top10)
    best_params = top10[0]

    params_ = individ_to_params(best_params)
    params_['n_feat'] = 2
    print(model_score_fitness(params=best_params))
    samples, labels = generated_dataset(params=params_)
    show_clusters(samples=samples, labels=labels)
    show_fitness_history(history=history)
