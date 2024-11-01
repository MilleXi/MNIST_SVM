from PIL import Image
import os
import sys
import numpy as np
import time
from sklearn import svm
import joblib
import warnings
warnings.filterwarnings('ignore')

def get_file_list(filepath):
    img_list=[]
    for i in os.listdir(filepath):
        for j in os.listdir(os.path.join(filepath,i)):
            if j.endswith('png'):
                img_list.append(os.path.join(filepath,i,j))
    return img_list

def get_img_name_str(imgPath):
    return imgPath.split(os.path.sep)[-1]

def img2vector(imgFile):
    img = Image.open(imgFile).convert('L')
    img_arr=np.array(img,'i')                       
    img_normalization=np.round(img_arr /255)          
    img_arr2=np.reshape(img_normalization,(1,-1))  
    return img_arr2

def read_and_convert(imgFilelist):
    dataLabel= []                
    dataNum =len(imgFilelist)
    dataMat = np.zeros((dataNum, 784))
    for i in range(dataNum):
        imgNameStr=imgFilelist[i]
        dataLabel.append(imgNameStr.split(os.path.sep)[1])
        dataMat[i,:] =img2vector(imgNameStr)
    # Sprint(dataLabel)
    return dataMat,dataLabel

def read_all_data():
    train_data_path = 'mnist_train'
    flist = get_file_list(train_data_path)
    # print(flist)
    dataMat,dataLabel = read_and_convert(flist)
    return dataMat,dataLabel

def create_svm(dataMat, dataLabel,path,decision='ovr') :
    clf=svm.SVC(decision_function_shape=decision)
    rf =clf.fit(dataMat, dataLabel)
    joblib.dump(rf, path) 
    return clf

if __name__=='__main__':
    print('正在运行模型请稍等')
    dataMat, dataLabel= read_all_data()    
    path = sys.path[0]
    model_path=os.path.join(path,r'svm.model')
    create_svm (dataMat,dataLabel, model_path, decision='ovr')
    print('模型训练存储完成')