# coding:utf-8

"""
聚类（非监督学习）
聚类的目的也是把数据分类，但是事先我是不知道如何去分的，
完全是算法自己来判断各条数据之间的相似性，相似的就放在一起。
在聚类的结论出来之前，我完全不知道每一类有什么特点，
一定要根据聚类的结果通过人的经验来分析，看看聚成的这一类大概有什么特点。
"""
import random
import numpy as np
import matplotlib.pyplot as plt

def loadDataSet():
    """
    1、获取数据
    :return: 数据集
    """
    dataSet = np.loadtxt("dataSet.csv")
    return dataSet

def initCentroids(dataSet, k):
    """
    2、从数据集中随机选取 k个数据（质点）返回（这里的 k可以通过肘部法则确定）
    :param dataSet:所有数据集
    :param k: k个质心点
    :return: 质心点
    """
    dataSet = list(dataSet)
    return random.sample(dataSet, k)

def calcuDistance(vec1, vec2):
    """
    计算两点的欧式距离方法
    :param vec1: 一个点
    :param vec2: 另一个点
    :return: 两点间的距离
    """
    return np.sqrt(np.sum(np.square(vec1 - vec2)))

def minDistance(dataSet, centroidList):
    """
    3、计算每个点到各个质点的距离，选取距离最近的质点，与该质点归为一簇类
    :param dataSet: 所有数据集
    :param centroidList: 质心点
    :return: 分类结果（字典）
    """
    # 创建一个空字典，保存簇类的结果
    clusterDict = dict()
    k = len(centroidList)
    for item in dataSet:
        vec1 = item
        flag = -1
        # float("inf")-->正无穷，初始化为最大值
        minDis = float("inf")
        for i in range(k):
            vec2 = centroidList[i]
            distance = calcuDistance(vec1, vec2)
            if distance < minDis:
                minDis = distance
                # 循环结束时，flag保存与当前item最近的簇标记
                flag = i
        if flag not in clusterDict.keys():
            clusterDict.setdefault(flag, [])
        # 加入相应的类别中
        clusterDict[flag].append(item)
    return clusterDict


def getVar(centroidList, clusterDict):
    """
    为什么要计算这个距离和？？？
    # 计算各簇集合间的均方误差
    # 将簇类中各个向量与质心点的距离累加求和
    :param centroidList: 质心点
    :param clusterDict: 分类结果（字典）
    :return: 距离和
    """
    sum = 0.0
    for key in clusterDict.keys():
        # 对应的质点
        vec1 = centroidList[key]
        distance = 0.0
        for item in clusterDict[key]:
            vec2 = item
            distance += calcuDistance(vec1, vec2)
        sum += distance
    return sum

def getCentroids(clusterDict):
    """
    4、重新计算 k个质心点
    :param clusterDict: 分类结果（字典）
    :return: 新的质心点
    """
    centroidList = []
    for key in clusterDict.keys():
        # mean() 求平均值
        # axis不设置值，对 m * n个数求均值，返回一个实数
        # axis = 0：压缩行，对各列求均值，返回 1 * n矩阵
        # axis = 1 ：压缩列，对各行求均值，返回 m * 1矩阵
        centroid = np.mean(clusterDict[key], axis=0)
        centroidList.append(centroid)
    return centroidList

def showCluster(centroidList, clusterDict):
    """
    展示聚类结果
    :param centroidList: 质心点
    :param clusterDict: 分类结果（字典）
    :return:
    """
    colorMark = ['or', 'ob', 'og', 'ok', 'oy', 'ow']  # 不同簇类标记，o表示圆形，另一个表示颜色
    centroidMark = ['dr', 'db', 'dg', 'dk', 'dy', 'dw']

    for key in clusterDict.keys():
        # 绘制质心点
        plt.plot(centroidList[key][0], centroidList[key][1], centroidMark[key], markersize=12)
        # 绘制其他数据点
        for item in clusterDict[key]:
            plt.plot(item[0], item[1], colorMark[key])
    plt.show()

# 没有进行聚类的散点图
def show_fig():
    dataSet = loadDataSet()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataSet[:, 0], dataSet[:, 1])
    plt.show()


def test_k_means():
    """

    :return:
    """
    # 加载数据
    dataSet = loadDataSet()
    # 获取 k个质点，k = 4
    centroidList = initCentroids(dataSet, 4)
    # 计算欧式，进行分类
    clusterDict = minDistance(dataSet, centroidList)
    newVar = getVar(centroidList, clusterDict)
    oldVar = 1 # 当两次聚类的误差小于某个值时，说明质心点基本确定

    times = 2
    while abs(newVar - oldVar) >= 0.00001:
        centroidList = getCentroids(clusterDict)
        clusterDict = minDistance(dataSet, centroidList)
        oldVar = newVar
        newVar = getVar(centroidList, clusterDict)
        times += 1
        showCluster(centroidList, clusterDict)




if __name__ == '__main__':
    show_fig()
    test_k_means()