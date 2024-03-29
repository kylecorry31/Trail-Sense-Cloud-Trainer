import numpy as np
import csv

import matplotlib.pyplot as plt
from numpy import random

rows = []
data_root = 'data'

with open('clouds.csv') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        rows.append(row)

inverse_cloud_type_map = [
    'Ci',
    'Cc',
    'Cs',
    'As',
    'Ac',
    'Ns',
    'Sc',
    'Cu',
    'St',
    'Cb'
]

size = len(inverse_cloud_type_map)

# Runs softmax on X
def predict(X, w):
    distances = np.zeros(shape=(w.shape[0]))
    for i in range(w.shape[0]):
        distances[i] = 1 / distance(X, w[i])
    exponents = np.exp(distances - np.max(distances))
    return [exponents / np.sum(exponents)]

def distance(X, sample):
    weights = np.array([[1] * len(sample)])
    weights = weights / np.sum(weights)
    return np.sum(np.abs(X - sample) * weights)

def train(X, y):
    averages = np.zeros(shape=(y.shape[1], X.shape[1]))
    counts = np.zeros(shape=(y.shape[1]))

    for i in range(len(X)):
        label = np.argmax(y[i])
        averages[label] += X[i]
        counts[label] += 1

    for i in range(len(averages)):
        if counts[i] > 0:
            averages[i] /= counts[i]

    return averages

# Calculates the cross-entropy loss
def fCE(X, w, y): # TODO: figure this out
    predictions = predict(X, w)
    n = y.shape[0]
    return np.sum(-y*np.log(predictions)) / n

# Calculates the cross-entropy loss gradient
def fCE_gradient(X, w, y):
    n = y.shape[0]
    predictions = predict(X, w)
    pred = predictions - y
    return X.T.dot(pred) / float(n)

def get_label(id):
    global size
    arr = [0] * size
    arr[id] = 1
    return arr

def get_data(row):
    return list(map(float, row[1:]))

def argmax(values):
    m = 0
    for i in range(0, len(values)):
        if values[i] > values[m]:
            m = i
    return m

def print_weights(weights):
    weight_strings = list(map(lambda x: "arrayOf(" + ", ".join(map(lambda s: str(s) + 'f', x)) + ")", weights))
    weight_arr = "arrayOf(" + ", ".join(weight_strings) + ")"
    f = open(data_root + "/averages.txt", "w")
    f.write(weight_arr)
    f.close()

def print_input(inputs):
    x_strings = list(map(lambda x: "arrayOf(" + ", ".join(map(lambda s: str(s) + 'f', x)) + ")", inputs))
    x_arr = "arrayOf(" + ", ".join(x_strings) + ")"
    f = open(data_root + "/input.txt", "w")
    f.write(x_arr)
    f.close()

def print_labels(labels):
    x_strings = list(map(lambda x: "arrayOf(" + ", ".join(map(str, x)) + ")", labels))
    x_arr = "arrayOf(" + ", ".join(x_strings) + ")"
    f = open(data_root + "/labels.txt", "w")
    f.write(x_arr)
    f.close()

X = np.array(list(map(get_data, rows)))
Y = np.array(list(map(lambda x: get_label(int(x[0])), rows)))

randomIndices = np.random.permutation(np.arange(X.shape[0]))

X = X[randomIndices]
Y = Y[randomIndices]

# TODO: Split into train and test data when there is enough samples

weights = train(X, Y)
print_weights(weights)

correct = 0

confusionMatrix = []
for i in range(0, size):
    row = []
    for j in range(0, size):
        row.append(0)
    confusionMatrix.append(row)

samples = [0] * size

for i in range(0, len(X)):
    prediction = predict(np.array([X[i]]), weights)[0]
    value = argmax(prediction)
    true_value = argmax(Y[i])
    samples[true_value] += 1
    confusionMatrix[true_value][value] += 1

    if value == true_value:
        correct += 1

print('     ', list(map(lambda x: f' {inverse_cloud_type_map[x]} ', range(0, size))))
for row in range(0, len(confusionMatrix)):
    print(f'{inverse_cloud_type_map[row]:5}', list(map(lambda x: "{:.2f}".format(x / samples[row]) if samples[row] > 0 else "0.00", confusionMatrix[row])))

print(correct / len(rows))