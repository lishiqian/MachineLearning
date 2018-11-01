# coding=utf8
from numpy import *
from os import listdir
import kNN


# 图片文件变向量
def fileToVector(filename):
    returnVec = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVec[0, 32 * i + j] = int(lineStr[j])
    return returnVec


def handWritingClassTest():
    # 读取训练样本
    hwLabels = []
    trainingFileList = listdir('data/trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNameStr = int(fileStr.split('_')[0])
        hwLabels.append(classNameStr)
        trainingMat[i, :] = fileToVector('data/trainingDigits/%s' % fileNameStr)

    # 读取测试样本
    testFileList = listdir('data/testDigits')
    mTest = len(testFileList)
    errorCount = 0.0
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNameStr = int(fileStr.split('_')[0])
        vectorUnderTest = fileToVector('data/testDigits/%s' % fileNameStr)
        classofierResult = kNN.kNN(vectorUnderTest, trainingMat, hwLabels, 5)
        print "the classofier came back with:%d,the real answer is:%d" % (classofierResult, classNameStr)

        if (classofierResult != classNameStr):
            errorCount += 1.0

    print "the total count rate is:%f" % errorCount
    print "the total error rate is:%f" % (errorCount / float(mTest))


handWritingClassTest()
