from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from generators.mdc import generated_dataset


def show_clusters(samples, labels):
    plt.scatter(samples[:, 0], samples[:, 1], c=labels, s=10)
    plt.show()


def dec_tree_score(dataset: Tuple) -> Tuple[float, float]:
    samples, labels = dataset

    features_train, features_test, target_train, target_test = \
        train_test_split(samples, labels, test_size=0.2)
    dt = DecisionTreeClassifier()

    dt.fit(features_train, target_train)

    train_score = dt.score(features_train, target_train)
    test_score = dt.score(features_test, target_test)

    return train_score, test_score


def log_reg_score(dataset: Tuple) -> Tuple[float, float]:
    samples, labels = dataset

    features_train, features_test, target_train, target_test = \
        train_test_split(samples, labels, test_size=0.2)

    target_train = np.ravel(target_train)
    target_test = np.ravel(target_test)
    log_reg = LogisticRegression()

    log_reg.fit(features_train, target_train)

    train_score = log_reg.score(features_train, target_train)
    test_score = log_reg.score(features_test, target_test)

    return train_score, test_score


if __name__ == '__main__':
    samples, labels = generated_dataset()
    train_score, test_score = dec_tree_score(dataset=(samples, labels))
    print(train_score)
    print(test_score)
