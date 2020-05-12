import cv2
import numpy as np 
import math 
import matplotlib.pyplot as plt

def createDateset():
    data = [23,12,3,54,89,2,43,7,45,26]
    label = [1,1,-1,-1,-1,1,1,-1,-1,1]
    return data,label 

# data,label = createDateset()
# print(data)
# print(label)

def calWeakClassifyOutput(input,weakClassify):
    # within then expected range, set 1, else set -1
    if weakClassify[0] == "left":
        #when it is left, if less than threshold value, output 1, else 0
        if input < weakClassify[1]:
            return 1
        elif input >= weakClassify[1]:
            return -1
    elif weakClassify[0] == "right":
        #when it is right, if greater than threshold value, output 1, else 0
        if input > weakClassify[1]:
            return 1
        elif input <= weakClassify[1]:
            return -1

# train weak classifier
def trainWeakClassifier(data,label,W):
    # to train classifiers, find a threshold value, split data into two parts, minimize classification error
    weakClassify = []
    # set every single data as threshold value, find the classification error rate, need cal len(data) times
    for m in data:
        for direction in ["left","right"]:
            i = 0
            error = 0
            for mm in data:
                if direction == "left":
                    # when left, if less than threshold value, label should be 1
                    # if label is -1, means wrong, need to increase weight
                    if mm < m and label[i] == -1:
                        error +=W[i]
                    elif mm >=m and label[i] == 1:
                        error +=W[i]
                if direction == "right":
                    # when right, if greater than threshold value, lable shpuld be 1
                    #if label is -1, means wrong
                    if mm > m and label[i] == -1:
                        error +=W[i]
                    elif mm <= m and label[i] == 1:
                        error +=W[i]
                i += 1
            weakClassify.append([direction,m,error[0]])
    
    # findout the weakClassifier with minimum error rate
    bestWeakClassifier = []
    for classifier in weakClassify:
        if not bestWeakClassifier:
            bestWeakClassifier = classifier
        else:
            if classifier[2] < bestWeakClassifier[2]:
                bestWeakClassifier = classifier
    return bestWeakClassifier

def adaboostTrain(desAccuracy,MaxWeakClassifierNum):
    #get train dataset
    data,label = createDateset()
    #initialize weights 1/n
    W = np.ones((len(data),1))/len(data)
    weakClassfiers = []
    accuracy = 0
    for num in range(MaxWeakClassifierNum):
        #train weak classifier
        weakclassfier = trainWeakClassifier(data,label,W)
        #update weights
        weight = 0.5*math.log(((1-weakclassfier[2])/weakclassfier[2]))
        print("weight",weight)
        print("weakClassifier",weakclassfier)
        weakclassfier.append(weight)
        weakClassfiers.append(weakclassfier)

        #update weights for train set
        midW = np.zeros(W.shape)
        for i in range(len(W)):
            midW[i] = W[i]*math.exp(-calWeakClassifyOutput(data[i],weakclassfier)*label[i]*weight)
        Zt = np.sum(midW)
        for i in range(len(W)):
            W[i] = midW[i]/Zt

        # check both left and right to see whether the output qualify
        i = 0
        accuracy = 0
        for d in data:
            result = 0
            for classfy in weakClassfiers:
                result += classfy[3]*calWeakClassifyOutput(d,classfy)
            if result > 0:
                result = 1
            else:
                result = -1
            if result == label[i]:
                accuracy += 1/len(data)
            i +=1
        print("accuracy ",accuracy)
        print("num ",num)
        if accuracy >= desAccuracy:
            break

adaboostTrain(0.98,10)






