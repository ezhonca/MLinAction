from numpy import *
mm = [1, 2, 3]
print(mm)
ff = array((1, 2, 3))
matric = ff * 2
print (matric)
group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
print(group.shape[0])
print(group.shape[1])
print(group.shape)
testMat = tile(mm,(group.shape[0],1))
print(testMat)
print(testMat.sum(axis=0).argsort())




def creatDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX,(dataSetSize,1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDisIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        label = labels[sortedDisIndicies[i]]
        classCount[label] = classCount.get(label,0)+1
    sortedClassCount = sorted(classCount.items(),key=lambda item:item[1],reverse=True)
    return sortedClassCount


group, labels = creatDataSet()
print(classify0([0,0],group,labels,3))

def file2matrix(filename):
    fr = open(filename)
    arrayOfLines = fr.readlines()
    numberOfLines = len(arrayOfLines)
    returnMat = zeros((numberOfLines,3))
    classLabelVectors=[]
    index= 0
    for line in arrayOfLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVectors.append(int(listFromLine[-1]))
        index +=1
    return returnMat,classLabelVectors




