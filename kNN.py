from numpy import *
mm = [1, 2, 3]
print(mm)
ff = array((1, 2, 3))
matric = ff * 2
print (matric)
group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
print(group.shape[0])
testMat = tile(mm,(group.shape[0],1))
print(testMat)

def creatDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

def classify(inX, dataSet, lables, k):
    dataSetSize = dataSet.shapr[0]


