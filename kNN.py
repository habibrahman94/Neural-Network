##kth Nearest Neighbor Algorithm
##Implemented on Python 3

import csv #Comma Separated Values
import random 
import math
import operator

##loadFile for open the dataset and dividing it to test and train data
def loadFile(fileName, splitter, trainingSet=[], testSet=[]):
    file_to_load = open(fileName, 'r') #Read the file
    lines = csv.reader(file_to_load)
    dataset = list(lines)
    for i in range(len(dataset)-1):
        for j in range(4):
            dataset[i][j] = float(dataset[i][j])
        if random.random()<splitter: #Compare the Random Value with the Splitter to differentiate
            trainingSet.append(dataset[i]) #Add to the List of trainingSet
        else:
            testSet.append(dataset[i])
    file_to_load.close() #A file should be close if its open after processing


##Euclidian Distance returns the distance between two vectors & Length is the dimension of the vector
def EuclidianDistance(instance1, instance2, length): 
    dist = 0
    for x in range(length):
        dist += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(dist)


##For a Single testData, compare with the TrainData to get the neighbors
def kthNeighbors(trainingSet, singleTestSet, K):
    dist = []
    length = len(singleTestSet)-1
    for j in range(len(trainingSet)-1):
        dd = EuclidianDistance(trainingSet[j], singleTestSet, length) ##Gets the Euclidian Distance between the TrainSet[j] and the TestSet
        dist.append((trainingSet[j], dd))
    dist.sort(key=operator.itemgetter(1)) ##Sort Based on The distance parameter
    neighbors=[]
    for i in range(K):
        neighbors.append(dist[i][0])
    return neighbors ##Returns the neighbors


##GetMajor from the Neighbors
def getMajor(neighbors):
    classVotes = {} ##Mapper for Several Keys
    for i in range(len(neighbors)):
        response = neighbors[i][-1] ##last element in the list
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

def main():
    trainingSet=[]
    testSet=[]
    loadFile('iris.data', 0.66, trainingSet, testSet)
    print('# of Train Set: '+repr(len(trainingSet)))
    print('# of Test Set: '+repr(len(testSet)))
    K=5 ##Kth 
    cnt=0
    for i in range(len(testSet)):
        neighbor = kthNeighbors(trainingSet, testSet[i], K) ##For each test Data
        result = getMajor(neighbor)
        if result == testSet[i][-1]:
            cnt+=1

    accuracy = ((cnt/len(testSet))*100) ## Number of Correct Set / # of testSet 
    print("Accuracy: "+ repr(accuracy)+'%')
    
main()
