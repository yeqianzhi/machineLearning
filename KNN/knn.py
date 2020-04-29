# coding:utf-8

import numpy as np
import matplotlib.pyplot as plt


def file2matrix(filepath="datingSet.csv"):
    """
    1、获取数据集，划分数据集
    :param filepath: 数据集路径
    :return: 数据集
    """
    # 1.1  获取数据
    dataSet = np.loadtxt(filepath)  # 获取 CSV文件数据
    # 1.2 获取属性值
    returnMat = dataSet[:, 0:-1]  # 获取 列索引0 至 倒数第2列 的 每一行数据
    # 1.3 获取标记值
    classlabelVector = dataSet[:, -1:]  # 获取 最后1列 数据
    return returnMat, classlabelVector  # 返回 属性值 以及 标记值

def autoNorm(dataSet):
    """
    2、数据归一化(属性)
    数据归一化是为了将不同表征的数据规约到相同的尺度内，
    常见的尺度范围有[-1, 1]，[0, 1]。
    A.min(0) : 返回A每一列最小值组成的一维数组；
    A.min(1)：返回A每一行最小值组成的一维数组；
    A.max(0)：返回A每一列最大值组成的一维数组；
    A.max(1)：返回A每一行最大值组成的一维数组；
    res = (x - min)/(max - min)
    :param dataSet: 数据集
    :return: 归一化后的数据集
    """
    minVal = dataSet.min(0)
    maxVal = dataSet.max(0)
    ranges = maxVal - minVal  # (max - min)

    # 创建一个ndarray，shape与 data相同，值填充为 0
    # normDataSet = np.zeros(dataSet.shape)
    # m, n = dataSet.shape  # m为行，n为列
    normDataSet = dataSet - minVal  # (x - min)
    normDataSet = normDataSet / ranges  # (x - min)/(max - min)
    # 返回归一化的值、每列最大值与最小值的差、每一列的最小值
    return normDataSet, ranges, minVal

def classify(inX, dataSet, labels, k):
    """
    3、定义knn算法分类器函数
    :param inX: 一条测试数据
    :param dataSet: 训练数据
    :param labels: 分类类别
    :param k: k值
    :return: 所属分类
    """
    # 3.1  计算欧式距离
    # |X| = 根号((x2-x1)^2 + (y2-y1)^2 + ...)
    dataSetSize = dataSet.shape[0]  # 行数，shape[1]列数
    # >>> a = array([1,2,3])
    # >>> c = tile(a, (2, 3))
    # >>> print c
    # [[1 2 3 1 2 3 1 2 3]
    #  [1 2 3 1 2 3 1 2 3]]
    # 为什么要进行这一步？计算这一条测试数据与所有训练数据的欧式距离
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)  # axis = 1 将一个矩阵的每一行向量相加
    distances = sqDistances ** 0.5  # 开根号

    # 3.2  排序
    sortedDistIndicies = distances.argsort()  # 排序并返回排序之前的 index

    classCount = {}
    # 3.3 取前 k个最近的样本的标记值
    for i in range(k):
        # 3.3.1 获取第i个（i<k）最近的样本的标记值
        voteIlabel = labels[sortedDistIndicies[i]][0]
        # Python 字典(Dictionary) get() 函数返回指定键的值，如果值不在字典中返回默认值
        # 3.3.2 将 key 为标记值，value 为该标记值的数量存入字典 classCount
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1 # default 0

    # 1、sorted(iterable, cmp=None, key=None, reverse=False)
    # iterable - - 可迭代对象。
    # cmp - - 比较的函数，这个具有两个参数，参数的值都是从可迭代对象中取出，此函数必须遵守的规则为，大于则返回1，小于则返回 - 1，等于则返回0。
    # key - - 主要是用来进行比较的元素，只有一个参数，具体的函数的参数就是取自于可迭代对象中，指定可迭代对象中的一个元素来进行排序。
    # reverse - - 排序规则，reverse = True
    # 降序 ， reverse = False
    # 升序（默认）。

    # 2、items()函数以列表返回可遍历的(键, 值)元组数组。

    # 3、lambda d: d[1] 在 items()的基础上根据 value值进行排序
    sortedClassCount = sorted(classCount.items(), key=lambda d: d[1], reverse=True)
    # 3.4 取标记值出现最多的标记值
    return sortedClassCount[0][0]

def datingClassTest(h=0.1):
    """
    4、定义测试算法的函数
    :param h: 测试集的占比
    :return: 打印出错误率
    """
    hoRatio = h
    # 4.1 分离 属性 datingDataMat 与 标记 datingLabels
    datingDataMat, datingLabels = file2matrix()
    # 4.2 获取归一化的值、每列最大值与最小值的差、每一列的最小值
    normMat, ranges, minVals = autoNorm(datingDataMat)
    # 4.3 m为行，n为列
    m, n = normMat.shape
    # 4.4 测试数据集的条数，默认取全部数据集的10%
    numTestVecs = int(m * hoRatio)  # 测试数据行数
    errorCount = 0  # 错误分类数

    # 用前 10%的数据做测试
    for i in range(numTestVecs):
        """
        def classify(inX, dataSet, labels, k):
            定义knn算法分类器函数
            :param inX: 一条测试数据
            :param dataSet: 训练数据
            :param labels: 分类类别
            :param k: k值
            :return: 所属分类
        """
        classifierResult = classify(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m],  4)
        if classifierResult != datingLabels[i]:
            errorCount += 1
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))

def show_fig():
    """
    5、简单分析数据，可以进行归纳偏好
    :return: 绘制每两个数据间的关系
    """
    data, cls = file2matrix()
    # figure() 创建一个图形实例
    fig = plt.figure()
    # add_subplot(111) 将图放在第 1行，第 1列，第一个位置
    fig.add_subplot(211)
    # scatter()绘制散点图
    plt.scatter(data[:, 0], data[:, 1], c=cls)

    fig.add_subplot(223)
    plt.scatter(data[:, 0], data[:, 2], c=cls)

    fig.add_subplot(224)
    plt.scatter(data[:, 1], data[:, 2], c=cls)
    # 横坐标
    plt.xlabel("playing game")
    # 纵坐标
    plt.ylabel("icm cream")
    # 绘制
    plt.show()

def classifypersion():
    """
    进行简单的预测
    :return:
    """
    resultList = ["none", 'not at all','in small doses','in large doses']
    # 模拟数据
    ffmiles = 15360
    playing_game = 8.545204
    ice_name = 1.340429

    datingDataMat, datingLabels = file2matrix()
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([ffmiles, playing_game, ice_name])
    # 预测数据归一化
    inArr = (inArr - minVals) / ranges
    classifierResult = classify(inArr, normMat, datingLabels, 3)
    print(resultList[int(classifierResult)])

if __name__ == '__main__':
    data, cls = file2matrix()
    print(data, cls)
    autoNorm(data)
    datingClassTest()
    classifypersion()
    show_fig()




