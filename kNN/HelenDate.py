# coding=utf8
# 海伦的约会
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
import kNN


# 从文件读取海伦约会数据
def fileToMatrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()  # 截取出说有的回车字符
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        if listFromLine[-1] == 'didntLike':
            classLabelVector.append(1)
        elif listFromLine[-1] == 'smallDoses':
            classLabelVector.append(2)
        elif listFromLine[-1] == 'largeDoses':
            classLabelVector.append(3)
        index += 1
    return returnMat, classLabelVector


dataSet, labels = fileToMatrix('data/datingTestSet.txt')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(dataSet[:, 1], dataSet[:, 2], 15.0 * array(labels), 15.0 * array(labels))
plt.show()


def datingClassTest():
    hoRatio = 0.10  # 测试数据占10%
    dataSet, labels = fileToMatrix('data/datingTestSet.txt')
    normMat, ranges, minValue = kNN.autoNorm(dataSet)
    m = normMat.shape[0]
    numTestVecs = int(hoRatio * m)  # 获取测试数据条数
    errorCount = 0.0
    for i in range(numTestVecs):
        # 前numTestVecs个数据作为测试集,后m-numTestVecs个数据作为训练集
        classofierResult = kNN.kNN(normMat[i, :], normMat[numTestVecs:m, :], labels[numTestVecs:m], 3)
        print "the classofier came back is%d,the real answer is:%d" % (classofierResult, labels[i])
        if (classofierResult != labels[i]):
            errorCount += 1.0
    print "the total error rate is:%f" % (errorCount / float(numTestVecs))


datingClassTest()
