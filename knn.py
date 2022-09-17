import numpy as np
import csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

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
    'Ac',
    'Ns',
    'Sc',
    'Cu',
    'St',
    'Cb'
]

size = len(inverse_cloud_type_map)

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

# 1, 5
X = np.array(list(map(lambda x: get_data(x), rows)))
Y = np.array(list(map(lambda x: get_label(int(x[0])), rows)))

# Split training data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train on all data
# X_train = X
# Y_train = Y
# X_test = X
# Y_test = Y

clf = KNeighborsClassifier(n_neighbors=7)
clf.fit(X_train, Y_train)

correct = 0
train_correct = 0

confusionMatrix = []
for i in range(0, size):
    row = []
    for j in range(0, size):
        row.append(0)
    confusionMatrix.append(row)

samples = [0] * size

for i in range(0, len(X_train)):
    prediction = clf.predict_proba([X_train[i]])
    for j in range(len(prediction)):
        prediction[j] = 1 - prediction[j][0][0]
    value = argmax(prediction)
    true_value = argmax(Y_train[i])
    samples[true_value] += 1
    confusionMatrix[true_value][value] += 1

    if value == true_value:
        train_correct += 1

for i in range(0, len(X_test)):
    prediction = clf.predict_proba([X_test[i]])
    for j in range(len(prediction)):
        prediction[j] = 1 - prediction[j][0][0]
    value = argmax(prediction)
    true_value = argmax(Y_test[i])

    if value == true_value:
        correct += 1

print('     ', list(map(lambda x: f' {inverse_cloud_type_map[x]} ', range(0, size))))
for row in range(0, len(confusionMatrix)):
    print(f'{inverse_cloud_type_map[row]:5}', list(map(lambda x: "{:.2f}".format(x / samples[row]) if samples[row] > 0 else "0.00", confusionMatrix[row])))

print()

for row in range(0, len(confusionMatrix)):
    print(f'{inverse_cloud_type_map[row]:5}', "{:.2f}".format(confusionMatrix[row][row] / samples[row] if samples[row] > 0 else 0.0))

print()
print("Train", train_correct / len(X_train), " (",  train_correct, "/", len(X_train), ")")
print("Test", correct / len(X_test), " (",  correct, "/", len(X_test), ")")
