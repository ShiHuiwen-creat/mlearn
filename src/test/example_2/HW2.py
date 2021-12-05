# import pandas as pd
# import sklearn as sl
# from IPython import display
# import numpy as np
# import math
# import csv
# from sklearn.tree import DecisionTreeClassifier
# l = np.zeros(3)
# print(l)
# train = pd.read_csv("data/train.csv")
# train_X = train.drop(columns=['income'])
# train_y = train['income']
# train_y = train_y.apply(lambda x:1 if x == ' >50K' else 0)
# test = pd.read_csv("data/test.csv")
# model = DecisionTreeClassifier()
# print(train_y.head(20))
# model.fit(train_X, train_y)
# prediction = model.predict(test)
# print(prediction)
import pandas as pd
import sklearn as sl
from IPython import display
import numpy as np
import math
import csv
def deal_data(train):
    # 先把特殊处理的列提取出来
    train_sex = train.sex
    train_income = train.income
    data_frame = []
    for i in train.columns:
        data_item = (i,train[i].dtype)
        data_frame.append(data_item)
    display.display(pd.DataFrame(data_frame))
    train = train.drop(['sex','income'],axis=1)
    #以 dtype 为分类器分类
    scatter_columns = [col for col in train.columns if train[col].dtype == 'object']
    scatter_data = train[scatter_columns]
    continuation_columns = [x for x in train.columns if x not in scatter_columns]
    continuation_data = train[continuation_columns]
    #独热编码
    scatter_data = pd.get_dummies(scatter_data)
    scatter_data['sex'] = train_sex.apply(lambda x:0 if x == ' Male' else 1)
    # 如果包含这个列，将其删除。
    if 'native_country_ Holand-Netherlands' in scatter_data.columns:
        print(scatter_data['native_country_ Holand-Netherlands'].value_counts())
        scatter_data = scatter_data.drop(['native_country_ Holand-Netherlands'],axis=1)
    train_data = pd.concat([scatter_data,continuation_data],axis=1)
# 归一化处理
    train_X = (train_data - train_data.mean()) / train_data.std()
    train_X = train_X.values
# 将结果标签修改成数值型
    train_y = train_income.apply(lambda x:0 if x == ' <=50K' else 1)
    train_y = train_y.values
    return train_X,train_y
def get_model(train_X,train_y):
    train_X_size = train_X.shape[0]
    class_0 = []
    class_1 = []
#将 class0 和 class1 分类
    for i in range(train_X_size):
        if train_y[i] == 1:
            class_1.append(list(train_X[i]))
        else:
            class_0.append(list(train_X[i]))
    sigma_0 = np.zeros((106,106))
    sigma_1 = np.zeros((106,106))
    class_0 = np.array(class_0)
    class_1 = np.array(class_1)
    mean_0 = np.mean(class_0,axis=0)
    mean_1 = np.mean(class_1,axis=0)
#求解 sigma

    for i in range(class_0.shape[0]):
        sigma_0 += np.dot(np.transpose([class_0[i] - mean_0]),[class_0[i] - mean_0])
    for i in range(class_1.shape[0]):
        sigma_1 += np.dot(np.transpose([class_1[i] - mean_1]),[class_1[i] - mean_1])
#放缩
    sigma_0 /= class_0.shape[0]
    sigma_1 /= class_1.shape[0]
    para_0 = class_0.shape[0] / train_X_size
    para_1 = class_1.shape[0] / train_X_size
    sigma = sigma_0*para_0+sigma_1*para_1
    N0 = class_0.shape[0]
    N1 = class_1.shape[0]
    print("Class 0:" ,N0)
    print("Class 1:",N1)
    #根据生成公式矩阵计算模型参数
    w = np.dot(mean_1-mean_0,np.linalg.pinv(sigma))
    b = np.dot(np.dot(mean_1,np.linalg.pinv(sigma)),np.transpose(mean_1)) * (-0.5)
    + np.dot(np.dot(mean_0,np.linalg.pinv(sigma)),np.transpose(mean_0)) * 0.5 +math.log(N1/N0)
    print(b)
    return w,b
def p(x,w,b):
#将模型抽象成线性模型
    result = np.dot(w,x) + b
    result *= -1
    # sigmiod函数
    p = 1 / (1+np.exp(result))
    return p
train = pd.read_csv("data\\train.csv")
train_X,train_y = deal_data(train)
w,b = get_model(train_X,train_y)
#读取 test 并预处理
test_data = pd.read_csv("data\\test.csv")
test_data['income'] = 0
test_X,test_y = deal_data(test_data)
model = p(test_X.T,w,b)
True_count = 0
False_count = 0
model_kind = []
for i in model:
    if i > 0.5:
        True_count += 1
        model_kind.append(' >50k')
    else:
        False_count += 1
        model_kind.append(' <=50k')
print(True_count,False_count)
# # 保存预测结果到文件中
source_file = 'data\\test.csv'
predict_file = pd.read_csv(source_file,low_memory=False)
predict_file['income'] = model_kind
predict_file['probability'] = model
#保存列到新的 csv 文件，index=0 表示不为每一行自动编号，header=1 表示行首有字段名称
predict_file.to_csv('data\\predict.csv', index=0, header=1)