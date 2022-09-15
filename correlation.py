import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('clouds.csv', names=['Type', 'Cover', 'EN', 'CON', 'Mean', 'VAR', 'Bias'])

print(df.describe())

print()
print("CORRELATION")
correlation = df.corr()
print(correlation.abs())