from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

from generators.mdc import generated_dataset


def show_clusters(samples, labels):
    plt.scatter(samples[:, 0], samples[:, 1], c=labels, s=10)
    plt.show()


# TODO: change return shape
def dec_tree_score(dataset: Tuple) -> Tuple[float, float]:
    samples, labels = dataset

    features = samples
    target = np.ravel(labels)

    dt = DecisionTreeClassifier()
    scores = cross_val_score(dt, features, target, cv=5)

    return scores.mean(), 0.5


def log_reg_score(dataset: Tuple) -> Tuple[float, float]:
    samples, labels = dataset

    features = samples
    target = np.ravel(labels)
    log_reg = LogisticRegression()

    scores = cross_val_score(log_reg, features, target, cv=5)

    return scores.mean(), 0.5


if __name__ == '__main__':
    samples, labels = generated_dataset()
    train_score, test_score = dec_tree_score(dataset=(samples, labels))
    print(train_score)
    print(test_score)
