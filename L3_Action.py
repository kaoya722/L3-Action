from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
import numpy as np

# 数据加载
data = pd.read_csv('D:/Program Files/Pycharm/Project/TEST/car_data.csv', encoding='gbk')
print(data)
train_x = data[["人均GDP", "城镇人口比重", "交通工具消费价格指数", "百户拥有汽车量"]]

# 数据规范化
min_max_scaler = preprocessing.MinMaxScaler()
train_x = min_max_scaler.fit_transform(train_x)
pd.DataFrame(train_x).to_csv('temp.csv', index=False)
print(train_x)

# 使用Kmeans进行聚类
kmeans = KMeans(n_clusters=4)
# Kmeans fit
predict_y = kmeans.fit_predict(train_x)

# 可视化打印
result = pd.concat((data, pd.DataFrame(predict_y)), axis=1)
print(result)
result.rename({0: u'聚类结果'}, axis=1, inplace=True)
print(result)
#result.to_csv("customer_cluster_result.csv", index=False)

#？？？？？？？为什么每次分类的运行结果会不一样？？？？？？？？？？？？

# kmeans手肘法统计不同K取值的误差平方和打印
import matplotlib.pyplot as plt

sse = []
for k in range(1, 11):
    # Kmeans进行聚类
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(train_x)
    # 计算inertia簇内误差平方和
    sse.append(kmeans.inertia_)
x = range(1, 11)
plt.xlabel('K')
plt.ylabel('SSE')
plt.plot(x, sse, 'o-')
plt.show()

