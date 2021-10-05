import numpy as np
import csv

rows = []

with open('clouds.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        rows.append(row)

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

# Runs softmax on X
def predict(X, w):
    z = np.dot(X, w)
    exponents = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exponents / np.sum(exponents, axis=1, keepdims=True)

def train_softmax(X, y):
    w = np.random.randn(X.shape[1], y.shape[1]) * 0.1
    T = 100
    n = 100
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
    arr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    arr[id] = 1
    return arr

def get_data(row):
    scale = 100.0
    return [
        int(row['CC']) / scale,
        int(row['CON']) / scale,
        int(row['EN']) / scale,
        int(row['ENT']) / scale,
        int(row['HOM']) / scale,
        int(row['LUM']) / scale,
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
    print("arrayOf(" + ", ".join(weight_strings) + ")")

X = np.array(list(map(get_data, rows)))

Y = np.array(list(map(lambda x: get_label(cloud_type_map[x['Type']]), rows)))

weights = train_softmax(X, Y)
print_weights(weights)

correct = 0

for row in rows:
    prediction = predict(np.array([get_data(row)]), weights)[0]
    value = argmax(prediction)
    if value == cloud_type_map[row['Type']]:
        correct += 1
    print(row['Type'], value, prediction[value])

print(correct / len(rows))