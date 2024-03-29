import numpy as np
import csv
from matplotlib import pyplot as plt

from numpy import random

rows = []

with open('clouds.csv') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        rows.append(row)

inverse_cloud_type_map = [
    'Ci',
    'Cc',
    'Cs',
    'As',
    'Ac', #
    'Ns',
    'Sc',
    'Cu', #
    'St',
    'Cb'
]

family_map = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# family_map = [0, 1, 0, 2, 1, 2, 1, 1, 2, 1]

size = max(family_map) + 1

def pca(X, k=2):
    Xtilde = (X.T - np.mean(X, axis=1)).T
    X2 = Xtilde.dot(Xtilde.T)
    values, vectors = np.linalg.eig(X2)
    d = vectors[:, np.flipud(np.argsort(values))][:, :k]
    print(d.T)
    return X.T.dot(d)

def get_data(row):
    return list(map(float, row[1:]))

# X = np.array(list(map(lambda x: float(x[1]), rows)))
# Y = np.array(list(map(lambda x: float(x[5]), rows)))
X = np.array(list(map(lambda x: get_data(x), rows)))
p = pca(X.T)
X = p[:, 0]
Y = p[:, 1]
colors = np.array(list(map(lambda x: 9 - family_map[int(x[0])], rows)))

plt.scatter(X, Y, c=colors, cmap='tab10')
plt.show()
