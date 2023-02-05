import matplotlib.pyplot as plt
import numpy as np
import random
import math


# 函数的作用是把一段代码封装成一个完整的功能,供别人去调用
# 函数有输入 输出

def scatter_cluster(data, cluster, center):
    if data.shape[1] != 2:
        raise ValueError('Only can scatter 2d data!')
    # 画样本点
    plt.scatter(data[:, 0], data[:, 1], c=cluster, alpha=0.8)
    mark = ['*r', '*b', '*g', '*k', '^b', '+b', 'sb', 'db', '<b', 'pb']
    # 画质心点
    for i in range(center.shape[0]):
        plt.plot(center[i, 0], center[i, 1], mark[i], markersize=20)
    plt.show()


def distance(point_1, point_2):
    x_1, y_1 = point_1[0], point_1[1]
    x_2, y_2 = point_2[0], point_2[1]
    distance = math.sqrt((x_1 - x_2) * (x_1 - x_2) + (y_1 - y_2) * (y_1 - y_2))
    return distance


def KMeans(data, k, max_iter=30):  # 等号代表缺省值，可传可不传
    # 首先拿到数据的相关参数（维度）
    n_samples, n_features = data.shape
    # 随机初始化簇中心
    indices = random.sample(range(n_samples), k)  # 随便找两个数字，数字代表的是样本的序号
    center = np.copy(data[indices])
    cluster = np.zeros(data.shape[0], dtype=np.int32)  # 定义一个空的array，为了记录属于哪个簇
    i = 1  # 定义一个值，作为当前迭代次数的计数
    while i <= max_iter:
        for n in range(n_samples):
            if n != indices[0] or indices[1]:
                dis_1 = distance(center[0], data[n])
                dis_2 = distance(center[1], data[n])
                if dis_1 <= dis_2:
                    cluster[n] = 0
                else:
                    cluster[n] = 1
        new_center = np.zeros((k, n_features))
        num_cls = np.zeros(k)
        for n in range(n_samples):
            cls = cluster[n]  # 获得当前点属于哪一类别
            num_cls[cls] += 1
            new_center[int(cls)] = new_center[int(cls)] + data[n]
        new_center = new_center / num_cls
        dis_between_centers = distance(new_center[0, :], center[0, :])
        dis_between_centers += distance(new_center[1, :], center[1, :])
        center = new_center
        if dis_between_centers < 0.005:
            break
        i += 1
    return new_center, cluster, i


if __name__ == '__main__':  # 入口函数
    # 如果想计算 k_means 需要样本
    n_samples = 50  # 样本数量
    n_features = 2  # 样本的特征维度
    max_iter = 40
    data = np.random.randn(n_samples, n_features)
    center, cluster, num_iter = KMeans(data, 2)
    scatter_cluster(data, cluster, center)
    k = 2
