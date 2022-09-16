import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('clouds.csv', names=['TYPE', 'NRBR', 'EN', 'CON', 'GLCM AVG', 'GLCM STDEV', 'BIAS'])

print(df.describe())

print()
print("CORRELATION")
correlation = df.corr()
print(correlation.abs())