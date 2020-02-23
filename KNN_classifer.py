import numpy as np
import mltools as ml
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("language_dataset.csv")
Y = []
X = []
for i in data:
    X.append(i[0])
    Y.append(i[1])

np.random.seed(0)

