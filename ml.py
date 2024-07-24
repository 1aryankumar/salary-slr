import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
database = pd.read_csv('Salary.csv')
x = database.iloc[:, :-1].values
y = database.iloc[:, -1].values
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)