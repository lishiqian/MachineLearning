# -*- coding: UTF-8 -*-
from numpy import *


"""
函数说明:创建实验样本

Parameters:
	无
Returns:
	postingList - 实验样本切分的词条
	classVec - 类别标签向量
Author:
	Jack Cui
Blog:
	http://blog.csdn.net/c406495762
Modify:
	2017-08-11
"""
def loadDataSet():
	postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],				#切分的词条
				['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
				['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
				['stop', 'posting', 'stupid', 'worthless', 'garbage'],
				['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
				['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
	classVec = [0,1,0,1,0,1]   			#类别标签向量，1代表侮辱性词汇，0代表不是
	return postingList, classVec		#返回实验样本切分的词条和类别标签向量

"""
函数说明:将切分的实验样本词条整理成不重复的词条列表，也就是词汇表

Parameters:
	dataSet - 整理的样本数据集
Returns:
	vocabSet - 返回不重复的词条列表，也就是词汇表
Author:
	Jack Cui
Blog:
	http://blog.csdn.net/c406495762
Modify:
	2017-08-11
"""
def createVocabList(dataSet):
	vocabSet = set([])
	for document in dataSet:
		vocabSet = vocabSet | set(document)
	return list(vocabSet)

"""
函数说明:根据vocabList词汇表，将inputSet向量化，向量的每个元素为1或0

Parameters:
	vocabList - createVocabList返回的列表
	inputSet - 切分的词条列表
Returns:
	returnVec - 文档向量,词集模型
Author:
	Jack Cui
Blog:
	http://blog.csdn.net/c406495762
Modify:
	2017-08-11
"""
def setOfWords2Vec(vocabList, inputSet):
	returnVec = [0] * len(vocabList)
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] = 1
		else:
			print "the word: %s is not in my vocabulaty" % word
	return returnVec

"""
函数说明:朴素贝叶斯分类器训练函数

Parameters:
	trainMatrix - 训练文档矩阵，即setOfWords2Vec返回的returnVec构成的矩阵
	trainCategory - 训练类别标签向量，即loadDataSet返回的classVec
Returns:
	p0Vect - 非的条件概率数组
	p1Vect - 侮辱类的条件概率数组
	pAbusive - 文档属于侮辱类的概率
Author:
	Jack Cui
Blog:
	http://blog.csdn.net/c406495762
Modify:
	2017-08-12
"""
def trainNB0(trainMatrix, trainCategory):
	numTrainDocs = len(trainMatrix) #计算训练的文档数目
	numWords = len(trainMatrix[0]) #计算每篇文档的词条数
	pAbusive = sum(trainCategory) / float(numTrainDocs) #文档属于侮辱类的概率
	p0Num = zeros(numWords); p1Num = zeros(numWords) #创建numpy.zeros数组
	p0Denom = 0.0; p1Denom = 0.0 #分母初始化为0.0
	for i in range(numTrainDocs):
		if trainCategory[i] == 1:  #统计属于侮辱类的条件概率所需的数据，即P(w0|1),P(w1|1),P(w2|1)···
			p1Num += trainMatrix[i]
			p1Denom += sum(trainMatrix[i])
		else:                       #统计属于非侮辱类的条件概率所需的数据，即P(w0|0),P(w1|0),P(w2|0）...
			p0Num += trainMatrix[i]
			p0Denom += sum(trainMatrix[i])
	p1Vect = p1Num / p1Denom
	p0Vect = p0Num / p0Denom
	return p0Vect,p1Vect,pAbusive  #返回属于侮辱类的条件概率数组，属于非侮辱类的条件概率数组，文档属于侮辱类的概率


"""
函数说明:朴素贝叶斯分类器分类函数

Parameters:
	vec2Classify - 待分类的词条数组
	p0Vec - 非侮辱类的条件概率数组
	p1Vec -侮辱类的条件概率数组
	pClass1 - 文档属于侮辱类的概率
Returns:
	0 - 属于非侮辱类
	1 - 属于侮辱类
Author:
	Jack Cui
Blog:
	http://blog.csdn.net/c406495762
Modify:
	2017-08-12
"""
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
	p1 = sum(vec2Classify * p1Vec) + log(pClass1)
	p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
	if p1 > p0:
		return 1
	else:
		return 0


"""
函数说明:测试朴素贝叶斯分类器

Parameters:
	无
Returns:
	无
Author:
	Jack Cui
Blog:
	http://blog.csdn.net/c406495762
Modify:
	2017-08-12
"""
def testingNB():
	listOfPosts, listClasses = loadDataSet()
	myVocabList = createVocabList(listOfPosts)
	trainMat = []
	for postinDoc in listOfPosts:
		trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
	p0v, p1v, pAb = trainNB0(array(trainMat),array(listClasses))
	testEntry = ['love', 'my', 'dalmation']
	thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
	print testEntry,'classified as', classifyNB(thisDoc,p0v,p1v,pAb)
	testEntry = ['stupid', 'garbage']
	thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
	print testEntry, 'classified as', classifyNB(thisDoc, p0v, p1v, pAb)


"""
函数说明:朴素贝叶斯池袋模型

Parameters:
	vocabList - createVocabList返回的列表
	inputSet - 切分的词条列表
Returns:
	returnVec - 文档向量,词集模型
Author:
	Jack Cui
Blog:
	http://blog.csdn.net/c406495762
Modify:
	2017-08-11
"""
def bagOfWords2VecMN(VocabList, inputSet):
    returnVec = [0] * len(VocabList)
    for word in inputSet:
        if word in VocabList:
            returnVec[VocabList.index(word)] += 1
    return returnVec

testingNB()


def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def spamTest():
    docList = []; classList = []; fullText = []
    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = range(50); testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []; trainClasses = []
    for docIndex in trainingSet:

        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList,docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
    print 'the error rate is:', float(errorCount/len(testSet))


spamTest()

# 统计单词出现最多的30个单词
def calcMostFreq(vocabList,fullText):
	import operator
	freqDict = {}
	for token in vocabList:
		freqDict[token] = fullText.count(token)
	sortedFreq = sorted(freqDict.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedFreq[:30]


def localWords(feed1, feed0):
	import feedparser
	docList = []; classList = []; fullText = []
	minLen = min(len(feed1['entries']), len(feed0['entries']))
	for i in range(minLen):
		wordList = textParse(feed1['entries'][i]['summary'])
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(1)
		wordList = textParse(feed0['entries'][i]['summary'])
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(0)
	vocabList = createVocabList(docList)
	# 计算出现次数最多的30个词 并且删除
	top30Words = calcMostFreq(vocabList, fullText)
	for pairW in top30Words:   # pariw : {'word' : 30}   pairW[0]: 'word'
		if pairW[0] in vocabList:
			vocabList.remove(pairW[0])
	# 随机抽取20条数据做测试数据
	trainingSet = range(2 * minLen); testSet = []
	for i in range(20):
		randIndex = int(random.uniform(0,len(trainingSet)))
		testSet.append(trainingSet[randIndex])
		del(trainingSet[randIndex])
	trainMat = []; trainClasses = []
	for docIndex in trainingSet:
		trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
		trainClasses.append(classList[docIndex])
	p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
	errorCount = 0
	for docIndex in testSet:
		wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
		if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
			errorCount += 1
	print 'the error rate is:', float(errorCount)/len(testSet)
	return vocabList, p0V, p1V


import feedparser
ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
vocabList, pSF, pNY = localWords(ny, sf)
vocabList, pSF, pNY = localWords(ny, sf)

# 最具表征性的词汇显示函数
def getTopWords(ny, sf):
	import operator
	vocabList, p0V, p1V = localWords(ny, sf)
	topNY = []; topSF = []
	for i in range(len(p0V)):
		if p0V[i] > -0.6 : topSF.append((vocabList[i]), p0V[i])
		if p1V[i] > -0.6 : topNY.append((vocabList[i]), p1V[i])
	sortedSF = sorted(topSF, key=lambda pair:pair[1],reverse=True)
	print 'SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**'
	for item in sortedSF:
		print item[0]
	sortedNY = sorted(topNY, key=lambda pair:pair[1],reverse=True)
	print 'NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**'
	for item in sortedNY:
		print item[0]