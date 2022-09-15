import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('clouds.csv', names=['Type', 'Cover', 'R', 'RB diff', 'EN', 'COR', 'Bias'])

correlation = df.corr()

print(correlation.abs())


# labels = [c[:2] for c in correlation.columns]
# fig = plt.figure(figsize=(12, 12))
# ax = fig.add_subplot(111)
# ax.matshow(correlation, cmap=plt.cm.RdYlGn)
# ax.set_xticks(np.arange(len(labels)))
# ax.set_yticks(np.arange(len(labels)))
# ax.set_yticklabels(labels)
# ax.set_xticklabels(labels)
# plt.show()
