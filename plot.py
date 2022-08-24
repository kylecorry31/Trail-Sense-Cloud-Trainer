import numpy as np
import csv
from matplotlib import pyplot as plt

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
    'As': 2,
    'Ac': 1,
    'Ns': 2,
    'Sc': 1,
    'Cu': 1,
    'St': 2,
    'Cb': 1
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

def pca(X, k=2):
    Xtilde = (X.T - np.mean(X, axis=1)).T
    X2 = Xtilde.dot(Xtilde.T)
    values, vectors = np.linalg.eig(X2)
    d = vectors[:, np.flipud(np.argsort(values))][:, :k]
    return X.T.dot(d)

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
        float(row['BSK'])
    ]

X = np.array(list(map(lambda x: get_data(x), rows)))
p = pca(X.T)
colors = np.array(list(map(lambda x: 9 - cloud_type_map[x['Type']], rows)))

plt.scatter(p[:, 0], p[:, 1], c=colors, cmap='tab10')
plt.show()
