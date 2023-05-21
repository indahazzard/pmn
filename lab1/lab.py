from skmultiflow.data import SEAGenerator
from skmultiflow.bayes import NaiveBayes 
from skmultiflow.trees import HoeffdingTree
from skelm import ELMClassifier
from skmultiflow.neural_networks import PerceptronMask
import matplotlib.pyplot as plt

stream = SEAGenerator(random_state=50, balance_classes=False, noise_percentage=0.87)

naive_bayes = NaiveBayes()
hoeffding_tree = HoeffdingTree()
ielm = ELMClassifier()
perceptron = PerceptronMask()

accuracy_naive_bayes = []
accuracy_hoeffding_tree = []
accuracy_ielm = []
accuracy_perceptron = []

nb_iters = 101
for i in range(nb_iters):
    X, y = stream.next_sample()

    naive_bayes.partial_fit(X, y)
    hoeffding_tree.partial_fit(X, y)
    ielm.partial_fit(X, y)
    perceptron.partial_fit(X, y, classes=stream.target_values)

    accuracy_naive_bayes.append(naive_bayes.score(X, y))
    accuracy_hoeffding_tree.append(hoeffding_tree.score(X, y))
    accuracy_ielm.append(ielm.score(X, y))
    accuracy_perceptron.append(perceptron.score(X, y))

iterations = [i for i in range(1, nb_iters)]


accuracy_hoeffding = [sum(accuracy_hoeffding_tree[:i])/len(accuracy_hoeffding_tree[:i]) 
            for i in range(1, nb_iters)]

accuracy_naive_b = [sum(accuracy_naive_bayes[:i])/len(accuracy_naive_bayes[:i]) 
            for i in range(1, nb_iters)]

accuracy_i = accuracy = [sum(accuracy_ielm[:i])/len(accuracy_ielm[:i]) for i in range(1, nb_iters)]

accuracy_p = [sum(accuracy_perceptron[:i])/len(accuracy_perceptron[:i]) for i in range(1, nb_iters)]

plt.plot(iterations, accuracy_naive_b, label='Bayes')
plt.plot(iterations, accuracy_hoeffding, label='HoeffdingTree')
plt.plot(iterations, accuracy_i, label='IELM')
plt.plot(iterations, accuracy_p, label='Perceptron')
plt.xlabel('Iterations')
plt.ylabel('Classification Accuracy')
plt.legend()
plt.savefig('graph.png')
plt.show()
