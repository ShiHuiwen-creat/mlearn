import pandas as pd
import sklearn as sl
from IPython import display
import numpy as np
import math
import csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

train = pd.read_csv('F:\\git\\python\\test-case-generator\\src\\test\\example_2\\data\\train.csv')
test = pd.read_csv('F:\\git\\python\\test-case-generator\\src\\test\\example_2\\data\\test.csv')
print(train.shape)

# 检查数据
def show_info(data, is_matrix_transpose=False) :
    # basic shape
    print('data shape is: {}   sample number {}   attribute number {}\n'.format(data.shape, data.shape[0], data.shape[1]))
    # attribute(key)
    print('data columns number {}  \nall columns: {}\n'.format(len(data.columns) ,data.columns))
    # value's null
    print('data all attribute count null:\n', data.isna().sum())
    # data value analysis and data demo
    if is_matrix_transpose:
        print('data value analysis: ', data.describe().T)
        print('data demo without matrix transpose: ', data.head().T)
    else:
        print('data value analysis: ', data.describe())
        print('data demo without matrix transpose: ', data.head())
train_X = train.drop(columns='income')
train_y = train['income']
train_y = train_y.apply(lambda x:1 if x == ' >50K' else 0)
# print(train_y.head(30))
data = pd.concat([train_X,test],axis=0)
print(data.head(5))
# print(train_X.columns)
# print(test.shape)
# print(data.sex.values)
# 热编码
data.sex = data.sex.apply(lambda x:1 if x == ' Male' else 0)
# print(data['sex'].head(20))
# print(data.head(20))

f, axes = plt.subplots(4, 4, figsize=(24, 18))
#
# for i, col in enumerate(train.columns):
#     train[col].value_counts().plot.bar(ax=axes[i // 4][i % 4])
#     plt.show()
# 字符集的col
categorical_cols = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'native_country']
# 数字集的col
numerical_cols = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']

# then encode categorical data
for col in categorical_cols:
    new = pd.get_dummies(data[col], prefix=col)
    data = pd.concat([data, new], axis=1)
    data = data.drop([col], axis=1)
# print(data.shape)
data = data.drop(['native_country_ Holand-Netherlands'], axis=1)
print(data.head(5))
# normalize numerical data
# 正规化
for col in numerical_cols:
    data[col] = (data[col] - data[col].mean()) / data[col].std()
# print(data)

# 下面使用决策树的model来做
# model = DecisionTreeClassifier()
# train_X = data.head(32561)
# test_X = data.tail(48842-32561)
#
# model.fit(train_X, train_y)
# pred = model.predict(test_X)
# print(pd.DataFrame(pred))

# 再将train.csv和test.csv分离出来
len = train_y.size
train = data.iloc[:len,:]
test = data.iloc[len:,:]

# 划分train_y中的0和1
class_0 = train.iloc[train_y[train_y == 0].index, :]
class_1 = train.iloc[train_y[train_y == 1].index, :]

# 导入一个将数组和矩阵拆分成随机训练和测试子集
from sklearn.model_selection import train_test_split


# cross_entropy交叉熵损失
def _cross_entropy_loss(y_pred, Y_label):
    '''
    y_pred [float]: output prediction(based on probabilistic)
    Y_label [bool]: label
    '''
    cross_entropy = -np.dot(Y_label, np.log(y_pred)) - np.dot((1 - Y_label), np.log(1 - y_pred))
    return cross_entropy


# sigmoid 激活函数
def _sigmoid(z):
    '''
    calculate probability
    '''
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - (1e-8))


# get probabilistiic 得到激活函数之后的概率
def get_prob(X, w, b):
    '''
    X: input data, shape = [batch_size, data_dimension]
    w: weight vector, shape = [data_dimension, ]
    b: bias, scalar
    '''

    return _sigmoid(np.matmul(X, w) + b)


# get prediction 得到预测
def infer(X, w, b):
    '''
    get prediction and transform data type
    '''
    return np.round(get_prob(X, w, b)).astype('int32')


# loss: call cross_entropy_loss
def _loss(y_pred, Y_label, lamda, w):
    return _cross_entropy_loss(y_pred, Y_label) + lamda * np.sum(np.square(w))


# gradient descent 梯度下降
def _gradient_regularization(X, Y_label, w, b):
    '''
    use cross_entropy to update weight and bias matrix
    add a lambda penalty term to avoid overfitting
    '''
    y_pred = get_prob(X, w, b)
    pred_error = Y_label - y_pred

    w_grad = -np.sum(pred_error * X.T, 1) + lamda * w
    b_grad = -np.sum(pred_error)
    return w_grad, b_grad


def _gradient(X, Y_label, w, b):
    '''
    use cross_entropy to update weight and bias matrix
    without a lambda penalty term
    '''
    y_pred = get_prob(X, w, b)
    pred_error = Y_label - y_pred
    # 填充矩阵
    # pred_error = np.pad(pred_error, ((0, 98), (0, 0)), 'constant', constant_values=(0, 0))

    w_grad = -np.sum(pred_error * X.T, 1)
    # print('w_grad')
    b_grad = -np.sum(pred_error)
    # print('b_grad')
    return w_grad, b_grad

# train_test_split
# 自己写封装函数
# artificial
def train_dev_split_(X, Y, dev_ratio):
    '''
    artificial split dataset into train_set and val_set
    '''
    train_size = int(X.shape[0] * (1 - dev_ratio))
    return X[:train_size], Y[:train_size], X[train_size:], Y[train_size:]
# 划分测试集和训练集
X_train, y_train, X_val, y_val = train_dev_split_(train,train_y,0.25)

def _shuffle(X, Y):
    randomize = np.arange(X.shape[0])
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])

def _accuracy(Y_pred, Y_label):
    acc = 1 - np.mean(np.abs(Y_pred - Y_label))
    return acc

# some prework

# init weight and bias with zero
w = np.zeros((X_train.shape[1], ))
b = np.zeros((1, ))

# hyperparam
max_iter = 100
batch_size = 8
learning_rate = 0.2
# regulariization
regularize = True
if regularize:
    lamda = 0.001
else:
    lamda = 0

loss_train = []
loss_validation = []
train_acc = []
dev_acc = []

step = 1
train_size = X_train.shape[0]
val_size = X_val.shape[0]
test_size = test.shape[0]
data_dim = X_train.shape[1]

X_train, Y_train = _shuffle(X_train.values, y_train.values)

for idx in range(int(np.floor(Y_train.shape[0] / batch_size))):
    X = X_train[idx * batch_size:(idx + 1) * batch_size]
    Y = Y_train[idx * batch_size:(idx + 1) * batch_size]
    # print("#****#")
    # 斜率
    w_grad, b_grad = _gradient(X, Y, w, b)
    # print(w_grad)
    # print(b_grad)

    w = w - learning_rate / np.sqrt(step) * w_grad
    b = b - learning_rate / np.sqrt(step) * b_grad

    step = step + 1


# normalize numerical data
def _normalize_column_normal(X, train=True, specified_column=None, X_mean=None, X_std=None):
    if train:
        if specified_column == None:
            specified_column = np.arange(X.shape[1])
            print("#****#")
        specified_column = np.array(specified_column)
        length = specified_column.shape[0]
        X_mean = np.reshape(np.mean(X[:, specified_column], 0), (1, length))
        X_std = np.reshape(np.std(X[:, specified_column], 0), (1, length))
    X[:, specified_column] = np.divide(np.subtract(X[:, specified_column], X_mean), X_std)

    return X, X_mean, X_std

col = [0, 1, 3, 4, 5, 7, 10, 12, 25, 26, 27, 28]

X_train, X_mean, X_std = _normalize_column_normal(X_train, specified_column=col)
# test = test.iloc[:, 1]
# w = np.pad(w,(0, 16281-106),'constant', constant_values=(0,0))
print(w.shape)
print(w)
w = w.reshape(106,1)
print(w)
print(b)
# test0 = np.array(test)
# test00 = test[0]
preds = infer(test.values, w, b)

print(preds)