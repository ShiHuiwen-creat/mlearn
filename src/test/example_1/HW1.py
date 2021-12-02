from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import csv

path = 'data/train.csv'
data_train = pd.read_csv(path)
# print(data_train)
data_train[data_train == 'NR'] = 0
data_train = data_train.iloc[:,3:]
# print(data_train.head(20))
path = 'data/test.csv'
data_test = pd.read_csv(path)
# print(data_test.head(30))
data = pd.DataFrame(data_test)
print(data.head(30))
print()
data_pm = []
# for line in data:
    # if data['AMB_TEMP'] == 'PM2.5' :


