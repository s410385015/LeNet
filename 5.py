import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import random
import cv2
import numpy as np

#model的架構，讀取model必須有其定義好的架構，不然無法讀取
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.ConvNet=nn.Sequential(
            nn.Conv2d(1,6,5,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        )
        self.Fc=nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )
    def forward(self,x):
        x=self.ConvNet(x)
        x=x.view(-1,16*5*5)
        x=self.Fc(x)
        
        return x

#取得已經train好的LeNet
def LoadModel():
    model=torch.load('LeNet.pkl')
    return model.cpu()


#取得training 跟 testing data
#在資料夾無data的時候，會自動從網路下載
def LoadData():
    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    trainSet = datasets.MNIST(root='MNIST', download=True, train=True)
    testSet = datasets.MNIST(root='MNIST', download=True, train=False)


    model_testSet = datasets.MNIST(root='MNIST', download=True, train=False,transform=transform)
    
    return trainSet,testSet,model_testSet


#5.1取得任意n張train data及其label
#TrainSet -> training的資料集
#Num -> 取得的張數 
#回傳 DataList -> (圖片,label)
def getTrainData(TrainSet,Num):
    DataList=[]
    nums = [x for x in range(len(TrainSet))]
    random.shuffle(nums)

    for i in range(Num):
        DataList.append(TrainSet[nums[i]])
    
    return DataList


#5.5 取得某張test data 及model predict 的結果
#TestSet ->  test的資料集
#model_testSet ->  model用的test資料及(基本上是一樣的，只是有經過normilze跟轉換成tensor
#model -> predict 用的model
#index  -> test資料集的 index 0~9999
#回傳  testData -> 圖片(不含label)  pred -> model predict的result
def getTestData(TestSet,model_testSet,model,index):
    testData=TestSet[index][0]

    
    x=model_testSet[index][0]
    x=x.view(1,1,28,28)
    model.eval()
    output=model(x)
    print(output)
    pred = output.argmax(dim=1, keepdim=True) 
    return testData,pred


model=LoadModel()

trainSet,testSet,model_testSet=LoadData()

print(model)


#5.1測試
'''
dataList=getTrainData(trainSet,10)
p1=np.array(dataList[0][0])
l1=dataList[0][1]

print(l1)
cv2.imshow('test',p1)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

#5.5
p1,l1=getTestData(testSet,model_testSet,model,5555)
p1=np.array(p1)

print(l1)
cv2.imshow('test',p1)
cv2.waitKey(0)
cv2.destroyAllWindows()




