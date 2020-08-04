import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from generators.mdc import generated_dataset


def show_clusters(samples, labels):
    plt.scatter(samples[:, 0], samples[:, 1], c=labels, s=10)
    plt.show()


if __name__ == '__main__':
    samples, labels = generated_dataset()
    # show_clusters(samples, labels)

    features_train, features_test, target_train, target_test = \
        train_test_split(samples, labels, test_size=0.2)
    dt = DecisionTreeClassifier()

    dt.fit(features_train, target_train)

    train_score = dt.score(features_train, target_train)
    test_score = dt.score(features_test, target_test)

    print(train_score)
    print(test_score)
