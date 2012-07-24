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
import matplotlib.pyplot as plt
from utils import *


class kNN:
    def __init__(self,k):
        self.k = k

    def createTestDataSet (self):
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
    def classify(self,dataSet,labels,inputVector):
        '''
        tile(input,A) repeats the input A times. We want the input to be replicated m times, but no replication along columns.
         So, tile args = (dataSet.numRows,1)
        '''
        diffMat = tile(inputVector, (dataSet.shape[0],1)) - dataSet
        #calculate Euclidean distance = sqrt((in.p1-data.p1)^2 +(in.p2-data.p2)^2+...)
        sqDiffMat = diffMat**2
        sqDistances = sqDiffMat.sum(axis=1)
        distances = sqDistances**0.5
        sortedDistIndices = distances.argsort()
        classCount={}
        #for each entry in the top k, find the label for that entry, and inc count of that label.
        for i in range(self.k):
            labelI = labels[sortedDistIndices[i]]
            classCount[labelI] = classCount.get(labelI,0) +1
        #find the most frequent label.
        sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1),reverse=True)
        return sortedClassCount[0][0]
    
if __name__=='__main__':
    kNNObject = kNN(4)
    #[group,labels] = kNNObject.createTestDataSet()
    utilityObject = utils()
    [group,labels] = utilityObject.file2Matrix('../datingTestSet.txt')
    classifiedLabel = kNNObject.classify(group,labels,[999,-222,0.1])
    print classifiedLabel
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(group[:,1],group[:,2])
    plt.show()