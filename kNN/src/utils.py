'''
Created on Jul 23, 2012

@author: chetan.kalyan
'''
from numpy import *

class utils:
    def file2Matrix(self,fileName):
        filePtr = open(fileName)
        listOfLines = filePtr.readlines()
        
        #find number of features in the file. Assumption: last column in each entry is the classifier label        
        testLine = listOfLines[0].strip()
        featureVector = testLine.split('\t')
        numFeatures = size(featureVector) - 1 #because last column is the label.
        
        featureMatrix = zeros((len(listOfLines),numFeatures))
        labelVector = []
        index = 0
        for line in listOfLines:
            featureVector = line.strip().split('\t')
            featureMatrix[index,:] = featureVector[0:-1]
            labelVector.append(featureVector[-1])
            index+= 1
        return featureMatrix,labelVector
        