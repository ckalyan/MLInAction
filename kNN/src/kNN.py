'''
Created on Jul 23, 2012

@author: chetan.kalyan

General approach:
1. Collect data
2. Prepare: numeric values are needed for all parameters for distance calculation.
3. Analyze: any method
4. Train: no training
5. Test: calculate P/R
'''
from numpy import *
import operator

def createDataSet ():
    group = array([[1,1.1],[1,1],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

'''
given a new data point Y,
for every point X in the dataset:
    calc dist between X and Y
sort the distances in increasing order
take the first 'k' items from this list
find the majority class among the items
return the majority class as the prediction for Y
'''
def classify(dataSet,labels,inputVector,k):
    dataSetSize = dataSet.shape[0]
    #tile(input,A) repeats the input A times
    diffMat = tile(inputVector, (dataSet.shape[0],1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndices = distances.argsort()
    classCount={}
    for i in range(k):
        labelI = labels[sortedDistIndices[i]]
        classCount[labelI] = classCount.get(labelI,0) +1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]