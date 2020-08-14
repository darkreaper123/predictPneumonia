import numpy as np
import os
import cv2
from PIL import Image
#Đọc ảnh từ folder với 1 số lượng nhất định
def readImage(array,folder,size):
 new_array=array
 count=0
 for filename in os.listdir(folder):
  img=Image.open(os.path.join(folder,filename)).convert('RGB')
  img=np.array(img)
  img=cv2.resize(src=img,dsize=(180,180))
  img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
  img=cv2.equalizeHist(img)
  new_array.append(img)
  count+=1
  if(count==size):
   break
 return new_array
#chuyển đổi dataset từ dạng [class1,class1,...,class1,class2,class2,...,class2] về dạng [class1,class2,...,class1,class2]
def balance(array):
 mid=int(np.array(array).shape[0]/2)
 new_array=[]
 for i in range(mid):
  new_array.append(array[i])
  new_array.append(array[i+mid])
 return np.array(new_array)
folder_image_normalChest=input("The folder contains normal chest images :")
folder_image_pneumoniaChest=input("The folder contains Pneumonia chest images :")
dataset=[]
dataset=readImage(dataset,folder_image_normalChest,1000)
dataset=readImage(dataset,folder_image_pneumoniaChest,1000)
dataset=balance(dataset)
#Dãn mỗi ảnh trong dataset về 1 mảng 1 chiều
dataset=dataset.reshape(dataset.shape[0],dataset.shape[1]*dataset.shape[2])
#Tạo label cho dataset
Y=[]
for i in range(dataset.shape[0]):
 if(i%2==0):
  Y.append(0)
 else:
  Y.append(1)
Y=np.array(Y).reshape(-1,1)
#Chia dataset thành dữ liệu để train và test
stringNumtrain="Size max is "+str(dataset.shape[0])+".The number of elements for training is "
numberForTrain=int(input(stringNumtrain))
X_train=dataset[:numberForTrain]
X_test=dataset[numberForTrain:]
Y_train=Y[:numberForTrain]
Y_test=Y[numberForTrain:]
#Tạo parameters
weight=np.random.rand(dataset.shape[1],1)*(10**-5)
bias=0.
stringLearning_rate="The learning rate is "
learning_rate=float(input(stringLearning_rate))
stringTrainLoop="The number of iterations is "
train_loop=int(input(stringTrainLoop))
#Activation function
def sigmoid(Z):
 array=[]
 for i in Z:
  if(i >= 0):
   array.append(1/(np.exp(-i)+1))
  else:
   array.append(np.exp(i)/(np.exp(i)+1))
 return np.array(array).reshape(-1,1)
#Vòng lặp cho thuật toán Logictics Regression
for i in range(train_loop):
 Y_predict=sigmoid(X_train.dot(weight)+bias)
 weight-=learning_rate*X_train.T.dot(Y_predict-Y_train)
 bias-=learning_rate*np.sum(Y_predict-Y_train,axis=0)
train_false=0
for score in abs(Y_predict-Y_train):
 if(score >=0.5):
  train_false+=1
trainScore=(Y_train.shape[0]-train_false)/Y_train.shape[0]
stringTrainAccuray="Train accuracy: "+str(trainScore*100)+"%."
print(stringTrainAccuray)
#Kiểm tra độ tốt của thuật toán bằng dữ liệu test
Y_predict_test=sigmoid(X_test.dot(weight)+bias)
test_false=0
for score in abs(Y_predict_test-Y_test):
 if(score >= 0.5):
  test_false+=1
testScore=(Y_test.shape[0]-test_false)/Y_test.shape[0]
stringTestAccuray="Test accuracy: "+str(testScore*100)+"%."
print(stringTestAccuray)
