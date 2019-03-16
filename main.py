from random import random
import math
from matplotlib import pyplot

'''
HELPER FUNCTIONS
'''
def target(row, theta):
    return ((float(row[0]) * theta[0]) + (float(row[1]) * theta[1]) + (float(row[2]) * theta[2]) + (float(row[3]) * theta[3]) + bias[0])

def sigmoid(target):
    return 1 / (1 + math.exp(-1 * float(target)))

def delta_theta(target, category, x):
    return 2 * (sigmoid(target) - float(category)) * (1 - sigmoid(target)) * sigmoid(target) * float(x)

def delta_bias(target, category):
    return 2 * (sigmoid(target) - float(category)) * (1 - sigmoid(target)) * sigmoid(target)

def prediction(sigmoid):
    return round(sigmoid)

def error(prediction, actual):
    return (prediction - actual) ** 2

data = open('iris_data.csv')

categories = {
    'Iris-setosa': '00',
    'Iris-versicolor': '01',
    'Iris-virginica': '10',
}

thetas = [([random()] * 4), ([random()] * 4)]

bias = [random()] * 2

learning_rate = 0.1

epoch = 100

'''
DATA PRE-PROCESSING
'''
data = data.read().split('\n')

'''
MODEL TRAINING
'''
accuracies = [0] * epoch
errors = [0] * epoch

for iteration in range(0, epoch):
    for row in data:
        row = data[0].split(',')

        targets = [target(row, thetas[0]), target(row, thetas[1])]
        sigmoids = [sigmoid(targets[0]), sigmoid(targets[1])]
        predictions = [prediction(sigmoids[0]), prediction(sigmoids[1])]

        for i in range(0, 2):
            category = float(categories[row[4]][i])
            correct = (predictions[i] == category)

            errors[iteration] += error(sigmoids[i], category)
            accuracies[iteration] += (1 if correct else 0)

            for j in range(0, 4):
                thetas[i][j] -= delta_theta(targets[i], category, row[j])
            bias[i] -= delta_bias(targets[i], category)

    errors[iteration] /= len(data)

pyplot.plot(range(1, epoch+1), errors)
pyplot.ylabel('Error')
pyplot.xlabel('Epoch')
pyplot.show()

pyplot.plot(range(1, epoch+1), accuracies)
pyplot.ylabel('Accuracy')
pyplot.xlabel('Epoch')
pyplot.show()
