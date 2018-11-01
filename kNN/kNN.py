# coding=utf8
# KNN.py
from numpy import *
import operator


def createDataSet():
    group = array(
        [[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])  # 我觉得可以这样理解，每一种方括号都是一个维度（秩），这里就是二维数组，最里面括着每一行的有一个方括号，后面又有一个，就是二维，四行
    labels = ['A', 'A', 'B', 'B']
    return group, labels


group, labels = createDataSet()


# K近邻算法
def kNN(inX, dataSet, labels,
        k):  # inX是你要输入的要分类的“坐标”，dataSet是上面createDataSet的array，就是已经有的，分类过的坐标，label是相应分类的标签，k是KNN，k近邻里面的k
    dataSetSize = dataSet.shape[0]  # dataSetSize是DataSet的行数，用上面的举例就是4行
    diffMat = tile(inX, (dataSetSize,
                         1)) - dataSet  # 前面用tile，把一行inX变成4行一模一样的（tile有重复的功能，dataSetSize是重复4遍，后面的1保证重复完了是4行，而不是一行里有四个一样的），然后再减去dataSet，是为了求两点的距离，先要坐标相减，这个就是坐标相减
    sqDiffMat = diffMat ** 2  # 上一行得到了坐标相减，然后这里要(x1-x2)^2，要求乘方
    sqDistances = sqDiffMat.sum(axis=1)  # axis=1是列相加，，这样得到了(x1-x2)^2+(y1-y2)^2
    distances = sqDistances ** 0.5  # 开根号，这个之后才是距离
    sortedDistIndicies = distances.argsort()  # argsort是排序，将元素按照由小到大的顺序返回下标，比如([3,1,2]),它返回的就是([1,2,0])
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,
                                                0) + 1  # get是取字典里的元素，如果之前这个voteIlabel是有的，那么就返回字典里这个voteIlabel里的值，如果没有就返回0（后面写的），这行代码的意思就是算离目标点距离最近的k个点的类别，这个点是哪个类别哪个类别就加1
    soredClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1),
                             reverse=True)  # key=operator.itemgetter(1)的意思是按照字典里的第一个排序，{A:1,B:2},要按照第1个（AB是第0个），即‘1’‘2’排序。reverse=True是降序排序
    return soredClassCount[0][0]  # 返回类别最多的类别


# 数据归一处理
def autoNorm(dataSet):
    minValue = dataSet.min(0)
    maxValue = dataSet.max(0)  # 参数0取列的最小值
    ranges = maxValue - minValue;
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minValue, (m, 1))  # 讲3*1矩阵变成 3 * m ，复制行
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minValue


print kNN([0, 0], group, labels, 2)
print kNN([2, 2], group, labels, 2)

print "hello world"
