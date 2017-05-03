#In k-fold cross-validation, the original sample is randomly partitioned into k equal 
#sized subsamples. Of the k subsamples, a single subsample is retained as the validation 
#data for testing the model, and the remaining k − 1 subsamples are used as training data. 
#The cross-validation process is then repeated k times (the folds), with each of the k 
#subsamples used exactly once as the validation data. The k results from the folds can 
#then be averaged to produce a single estimation. The advantage of this method over 
#repeated random sub-sampling (see below) is that all observations are used for both 
#training and validation, and each observation is used for validation exactly once. 
#10-fold cross-validation is commonly used,[6] but in general k remains an unfixed 
#parameter.

##From Wikipedia


import csv #Comma Separated Values
import random 
import math
import operator

##loadFile for open the dataset and dividing it to test and train data
def loadFile(fileName, set1=[], set2=[], set3=[]):
    file_to_load = open(fileName, 'r') #Read the file
    lines = csv.reader(file_to_load)
    data = list(lines)
    random.shuffle(data) ##Shuffled for random
    dataset = [x for x in data if x != []] ##Shuffle sometimes adds empty list, so it should removed
            
    for i in range(len(dataset)-1):
        for j in range(4):
            dataset[i][j] = float(dataset[i][j])
        if i<50:
            set1.append(dataset[i])
        elif i<100:
            set2.append(dataset[i])
        else:
            set3.append(dataset[i])
    
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
    set1=[]
    set2=[]
    set3=[]
    loadFile('iris.data', set1,set2,set3)
    
    K=3 ##Kth Nearest Neighbors
    
    ##set3 as testSet && set1 & set2 as trainSet
    cnt=0
    for i in range(len(set3)):
        neighbor = kthNeighbors(set1+set2, set3[i], K) ##For each test Data
        
        result = getMajor(neighbor)
        
        if result == set3[i][-1]:
            cnt+=1

    accuracy1 = ((cnt/len(set3))*100) ## Number of Correct Set / # of testSet 
    
    ##set2 as testSet && set1 & set3 as trainSet            
    cnt=0
    for i in range(len(set2)):
        neighbor = kthNeighbors(set1+set3, set2[i], K) ##For each test Data
        result = getMajor(neighbor)
        if result == set2[i][-1]:
            cnt+=1
    
    accuracy2 = ((cnt/len(set2))*100)
    
    ##set1 as testSet && set3 & set2 as trainSet
    cnt=0
    for i in range(len(set1)):
        neighbor = kthNeighbors(set2+set3, set1[i], K) ##For each test Data
        result = getMajor(neighbor)
        if result == set1[i][-1]:
            cnt+=1
    
    accuracy3 = ((cnt/len(set1))*100)
    
    ##Average Accuracy
    print("Accuracy: "+repr(((accuracy1+accuracy2+accuracy3)/3)))
    
main()
