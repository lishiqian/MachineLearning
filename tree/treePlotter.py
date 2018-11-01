# coding=utf8
import matplotlib.pyplot as plt
import trees

# 定义文本框和箭头样式
descisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction', xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


# 节点画法
# def createPlot():
#     fig = plt.figure(1,facecolor='white')
#     fig.clf()
#     createPlot.ax1 = plt.subplot(111,frameon=False)
#     plotNode("a decision node",(0.5,0.1),(0.1,0.5),descisionNode)
#     plotNode('a leaf node',(0.8,0.1),(0.3,0.8),leafNode)
#     plt.show()
#
# createPlot()


# 获取叶子节点
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == "dict":
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


# 获取树的高度
def getTreeDept(myTree):
    maxDept = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == "dict":
            thisDept = 1 + getNumLeafs(secondDict[key])
        else:
            thisDept = 1
        if thisDept > maxDept:
            maxDept = thisDept
    return maxDept


# 在父子节点间填充文本
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)


# 绘制决策树
def plotTree(myTree, parentPt, nodeText):
    numLeafs = getNumLeafs(myTree)
    dept = getTreeDept(myTree)
    firstStr = myTree.keys()[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeText)
    plotNode(firstStr, cntrPt, parentPt, descisionNode)

    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == "dict":
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))

    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


# 创建图形函数
def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDept(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW;
    plotTree.yOff = 1.0;
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()


def retrieveTree(i):
    listOfTrees = [
        {'no surfacing': {0: 'no', 1: {'filppers': {0: 'no', 1: 'yes'}}}}
        , {'no surfacing': {0: 'no', 1: {'filppers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'yes'}}}}]
    return listOfTrees[i]


myTree = retrieveTree(1)
print myTree
print getTreeDept(myTree)
print getNumLeafs(myTree)
createPlot(myTree)

myDat, labels = trees.createDataSet()
print labels
myTree = retrieveTree(0)
print myTree
print trees.classify(myTree, labels, [1, 0])
print trees.classify(myTree, labels, [1, 1])
