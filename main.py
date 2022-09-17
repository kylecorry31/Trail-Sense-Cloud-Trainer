import numpy as np
import csv

from numpy import random
from sklearn.model_selection import train_test_split

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
    z = np.dot(X, w)
    exponents = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exponents / np.sum(exponents, axis=1, keepdims=True)

def train_softmax(X, y):
    w = np.random.randn(X.shape[1], y.shape[1]) * 0.1
    T = 1000
    n = 2
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
    f = open(data_root + "/weights.txt", "w")
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

# Split training data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train on all data
# X = X[randomIndices]
# Y = Y[randomIndices]
# X_train = X
# Y_train = Y
# X_test = X
# Y_test = Y

# TODO: Split into train and test data when there is enough samples

weights = train_softmax(X_train, Y_train)
print_weights(weights)
print_input(X)
print_labels(Y)

correct = 0
train_correct = 0

confusionMatrix = []
for i in range(0, size):
    row = []
    for j in range(0, size):
        row.append(0)
    confusionMatrix.append(row)

samples = [0] * size

for i in range(0, len(X_test)):
    prediction = predict(np.array([X_test[i]]), weights)[0]
    value = argmax(prediction)
    true_value = argmax(Y_test[i])
    samples[true_value] += 1
    confusionMatrix[true_value][value] += 1

    if value == true_value:
        correct += 1

for i in range(0, len(X_train)):
    prediction = predict(np.array([X_train[i]]), weights)[0]
    value = argmax(prediction)
    true_value = argmax(Y_train[i])

    if value == true_value:
        train_correct += 1

print('     ', list(map(lambda x: f' {inverse_cloud_type_map[x]} ', range(0, size))))
for row in range(0, len(confusionMatrix)):
    print(f'{inverse_cloud_type_map[row]:5}', list(map(lambda x: "{:.2f}".format(x / samples[row]) if samples[row] > 0 else "0.00", confusionMatrix[row])))


print()

for row in range(0, len(confusionMatrix)):
    print(f'{inverse_cloud_type_map[row]:5}', "{:.2f}".format(confusionMatrix[row][row] / samples[row] if samples[row] > 0 else 0.0))

print()
print("Train", train_correct / len(X_train), " (",  train_correct, "/", len(X_train), ")")
print("Test", correct / len(X_test), " (",  correct, "/", len(X_test), ")")
