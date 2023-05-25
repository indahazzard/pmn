import numpy as np
from sklearn.base import BaseEstimator
from modAL.utils.data import modALinput
from modAL.utils.selection import multi_argmax

from modAL.models import ActiveLearner, Committee
from modAL.uncertainty import uncertainty_sampling, entropy_sampling

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from matplotlib import pyplot as plt
from modAL.density import information_density

n_initial = 150

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)

initial_idx = np.random.choice(range(len(X_train)), size=n_initial, replace=False)

X_initial, y_initial = X_train[initial_idx], y_train[initial_idx]
X_pool, y_pool = np.delete(X_train, initial_idx, axis=0), np.delete(y_train, initial_idx, axis=0)

learner = ActiveLearner(
    estimator=RandomForestClassifier(),
    query_strategy=uncertainty_sampling,
    X_training=X_initial, y_training=y_initial
)


def information_density_strategy(classifier: BaseEstimator, X: modALinput, n_instances=1, random_tie_break = False, **uncertainty_measure_kwargs):
    
    density = information_density(X, 'cosine')
    
    query_idx = multi_argmax(density, n_instances=n_instances)

    return query_idx



n_queries = 10

accuracy_scores = [learner.score(X_test, y_test)]

for i in range(n_queries):
    query_idx, query_inst = learner.query(X_pool)
    with plt.style.context('seaborn-white'):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title('Digit to label')
        plt.imshow(query_inst.reshape(8, 8))
        plt.subplot(1, 2, 2)
        plt.title('Accuracy of your model')
        plt.plot(range(i+1), accuracy_scores)
        plt.scatter(range(i+1), accuracy_scores)
        plt.xlabel('number of queries')
        plt.ylabel('accuracy')
        plt.show() 
    print("Which digit is this?")
    y_new = np.array([int(input())], dtype=int)
    learner.teach(query_inst.reshape(1, -1), y_new)
    X_pool, y_pool = np.delete(X_pool, query_idx, axis=0), np.delete(y_pool, query_idx, axis=0)
    accuracy_scores.append(learner.score(X_test, y_test))


with plt.style.context('seaborn-white'):
    plt.figure(figsize=(10, 5))
    plt.title('Accuracy of the classifier during the active learning')
    plt.plot(range(n_queries+1), accuracy_scores)
    plt.scatter(range(n_queries+1), accuracy_scores)
    plt.xlabel('number of queries')
    plt.ylabel('accuracy')
    plt.show()


learner_information_density = ActiveLearner(
    estimator=RandomForestClassifier(),
    query_strategy=information_density_strategy,
    X_training=X_initial, y_training=y_initial
)
score_info_denstiry = [learner_information_density.score(X_test, y_test)]
for i in range(n_queries):
    query_idx, query_inst = learner_information_density.query(X_pool)
    with plt.style.context('seaborn-white'):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title('Digit to label')
        plt.imshow(query_inst.reshape(8, 8))
        plt.subplot(1, 2, 2)
        plt.title('Accuracy of your model')
        plt.plot(range(i+1), score_info_denstiry)
        plt.scatter(range(i+1), score_info_denstiry)
        plt.xlabel('number of queries')
        plt.ylabel('accuracy')
        plt.show() 
    print("Which digit is this?")
    y_new = np.array([int(input())], dtype=int)
    learner_information_density.teach(query_inst.reshape(1, -1), y_new)
    X_pool, y_pool = np.delete(X_pool, query_idx, axis=0), np.delete(y_pool, query_idx, axis=0)
    score_info_denstiry.append(learner_information_density.score(X_test, y_test))


with plt.style.context('seaborn-white'):
    plt.figure(figsize=(10, 5))
    plt.title('Accuracy of the classifier during the active learning by information_density')
    plt.plot(range(n_queries+1), score_info_denstiry)
    plt.scatter(range(n_queries+1), score_info_denstiry)
    plt.xlabel('number of queries')
    plt.ylabel('accuracy')
    plt.show()


learners = [ActiveLearner(estimator=RandomForestClassifier(), query_strategy=entropy_sampling, X_training=X_initial, y_training=y_initial),
            ActiveLearner(estimator=RandomForestClassifier(), query_strategy=information_density_strategy, X_training=X_initial, y_training=y_initial)]

committee = Committee(
        learner_list=learners,
        query_strategy=entropy_sampling
        )

score_committee = [committee.score(X_test, y_test)]
for i in range(n_queries):
    query_idx, query_inst = committee.query(X_pool)
    with plt.style.context('seaborn-white'):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title('Digit to label')
        plt.imshow(query_inst.reshape(8, 8))
        plt.subplot(1, 2, 2)
        plt.title('Accuracy of your model')
        plt.plot(range(i+1), score_committee)
        plt.scatter(range(i+1), score_committee)
        plt.xlabel('number of queries')
        plt.ylabel('accuracy')
        plt.show() 
    print("Which digit is this?")
    y_new = np.array([int(input())], dtype=int)
    committee.teach(query_inst.reshape(1, -1), y_new)
    X_pool, y_pool = np.delete(X_pool, query_idx, axis=0), np.delete(y_pool, query_idx, axis=0)
    score_committee.append(committee.score(X_test, y_test))


with plt.style.context('seaborn-white'):
    plt.figure(figsize=(10, 5))
    plt.title('Accuracy of the classifier during the active learning by committee')
    plt.plot(range(n_queries+1), score_committee)
    plt.scatter(range(n_queries+1), score_committee)
    plt.xlabel('number of queries')
    plt.ylabel('accuracy')
    plt.show()

