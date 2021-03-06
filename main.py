import numpy as np
import csv

from numpy import random

rows = []

with open('clouds.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        rows.append(row)

size = 10

cloud_type_map = {
    'Ci': 0,
    'Cc': 1,
    'Cs': 2,
    'As': 3,
    'Ac': 4,
    'Ns': 5,
    'Sc': 6,
    'Cu': 7,
    'St': 8,
    'Cb': 9
}

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

# Runs softmax on X
def predict(X, w):
    z = np.dot(X, w)
    exponents = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exponents / np.sum(exponents, axis=1, keepdims=True)

def train_softmax(X, y):
    w = np.random.randn(X.shape[1], y.shape[1]) * 0.1
    T = 1000
    n = 5
    N = X.shape[0]
    epsilon = 0.1
    randomIndices = np.random.permutation(np.arange(N))
    randX = X[randomIndices]
    randY = y[randomIndices]
    for i in range(T):
        for r in range(int(np.ceil(N / float(n)))):
            batch = randX[r*n:r*n + n]
            batchY = randY[r*n:r*n + n]
            gradient = fCE_gradient(batch, w, batchY)
            w = w - epsilon * gradient
    return w

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
    return [
        float(row['CC']),
        float(row['R']),
        float(row['B']),
        float(row['RG']),
        float(row['RB']),
        float(row['GB']),
        float(row['EN']),
        float(row['ENT']),
        float(row['CON']),
        float(row['HOM']),
        float(row['BSTD']),
        float(row['BSK']),
        1
    ]

def argmax(values):
    m = 0
    for i in range(0, len(values)):
        if values[i] > values[m]:
            m = i
    return m

def print_weights(weights):
    weight_strings = list(map(lambda x: "arrayOf(" + ", ".join(map(lambda s: str(s) + 'f', x)) + ")", weights))
    weight_arr = "arrayOf(" + ", ".join(weight_strings) + ")"
    f = open("/data/weights.txt", "w")
    f.write(weight_arr)
    f.close()

def print_input(inputs):
    x_strings = list(map(lambda x: "arrayOf(" + ", ".join(map(lambda s: str(s) + 'f', x)) + ")", inputs))
    x_arr = "arrayOf(" + ", ".join(x_strings) + ")"
    f = open("/data/input.txt", "w")
    f.write(x_arr)
    f.close()

def print_labels(labels):
    x_strings = list(map(lambda x: "arrayOf(" + ", ".join(map(str, x)) + ")", labels))
    x_arr = "arrayOf(" + ", ".join(x_strings) + ")"
    f = open("/data/labels.txt", "w")
    f.write(x_arr)
    f.close()

X = np.array(list(map(get_data, rows)))
Y = np.array(list(map(lambda x: get_label(cloud_type_map[x['Type']]), rows)))

randomIndices = np.random.permutation(np.arange(X.shape[0]))

X = X[randomIndices]
Y = Y[randomIndices]

# TODO: Split into train and test data when there is enough samples

weights = train_softmax(X, Y)
print_weights(weights)
print_input(X)
print_labels(Y)

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