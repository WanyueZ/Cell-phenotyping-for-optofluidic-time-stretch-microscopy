#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function
from skimage import feature as ft
import math
from skimage import data, io, filters
from PIL import Image
import numpy as np
import os
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.svm import SVC
import time
import matplotlib.pyplot as plt
import skimage
from skimage import transform,data
from xgboost import XGBClassifier
import glob, sys, threading


import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, advanced_activations, Add, Lambda
from keras.layers import MaxPooling2D, Input, ZeroPadding2D, Reshape,Conv2DTranspose,concatenate
#from tensorflow.python.keras.layers import 
from keras.layers.advanced_activations import PReLU
import tensorflow as tf
from keras.models import load_model
from keras import optimizers
from keras import losses

from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
import argparse
from keras.losses import mean_absolute_error, mean_squared_error

import scipy.io
from scipy import ndimage, misc
import re
import csv

from keras.layers import Flatten,BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.merge import add
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

import six
from keras.regularizers import l2
from keras.utils import to_categorical
import matplotlib.pyplot as plt

#import numba
import gc
from keras.utils import multi_gpu_model
import random


# In[2]:


path_group_1="/Volumes/Seagate Expansion Drive/cells/Group1/"
path_group_2="/Volumes/Seagate Expansion Drive/cells/Group2/"
IMG_SIZE = 191
Epochs = 50
BATCH_SIZE = 128


# In[3]:


def get_image_list(data_path):
    l = glob.glob(os.path.join(data_path,"*"))
    l = [f for f in l if  '.tif' in  os.path.basename(f)]
    random.shuffle(l)
    #print(len)
    return l

def get_image(list_group,path_group,IMG_SIZE):
    i=0
    g1=np.zeros((len(list_group),IMG_SIZE*IMG_SIZE))
    g2=np.zeros((len(list_group),IMG_SIZE,IMG_SIZE))
    for filename in list_group:
        img_g2 = np.array(Image.open(os.path.join(path_group,filename)))
        img_g2 = img_g2[0:IMG_SIZE, 0:IMG_SIZE]
        img_g1 = np.reshape(img_g2,(1,IMG_SIZE*IMG_SIZE))
        g1[i, :] = img_g1
        g2[i, :, :] = img_g2
        i+=1
    return g1,g2


def DBSC(g1):
    pca = PCA(n_components = 500)
    model = pca.fit(g1)
    g1_pca = model.transform(g1)
    myarray = g1_pca
    myarray = myarray/np.max(myarray)
    modle = DBSCAN(eps = 0.44,min_samples = math.ceil(g1.shape[0]/2300))
    result_DBSC = modle.fit(myarray)
    labels = result_DBSC.labels_
    g1_pop_array = np.where(labels != 0)
    return g1_pop_array[0]

#@jit
def HOG_Gray(img_train):
    g1_hoghist=np.zeros((img_train.shape[0],5625+30))
    for j in range(img_train.shape[0]):
        img_g = img_train[j, :, :]
        img_hog = ft.hog(img_g, orientations=9, pixels_per_cell=(20, 20), cells_per_block=(5, 5), block_norm='L1', visualize=False, transform_sqrt=False, feature_vector=True)
        g1_hoghist[j, 0:5625]=img_hog
        hist2 = np.histogram(img_g, 30)
        g1_hoghist[j, 5625:5625+30]=hist2[0]
    return g1_hoghist


# In[9]:


###  data loading
list_group_1 = get_image_list(path_group_1)
list_group_2 = get_image_list(path_group_2)
img_1_1D, img_1_2D = get_image(list_group_1,path_group_1,IMG_SIZE)
img_2_1D, img_2_2D = get_image(list_group_2,path_group_2,IMG_SIZE)
img_train_1_1D = img_1_1D[0:round(len(list_group_1)/2),:]
img_train_1_2D = img_1_2D[0:round(len(list_group_1)/2),:,:]
img_train_2_1D = img_2_1D[0:round(len(list_group_2)/2),:]
img_train_2_2D = img_2_2D[0:round(len(list_group_2)/2),:,:]
img_test_1 = img_1_2D[round(len(list_group_1)/2):,:,:]
img_test_2 = img_2_2D[round(len(list_group_2)/2):,:,:]
img_test=np.vstack((img_test_1,img_test_2))
del img_1_1D, img_1_2D,img_2_1D, img_2_2D
gc.collect()
time.sleep(0.5)
pass


# In[10]:


###DBSCAN outlier detection

pop_array_1=DBSC(img_train_1_1D)
pop_array_2=DBSC(img_train_2_1D)

img_train_1=np.delete(img_train_1_2D, pop_array_1, axis=0)
img_train_2=np.delete(img_train_2_2D, pop_array_2, axis=0)

img_train=np.vstack((img_train_1,img_train_2))

label_train=np.vstack((np.zeros((img_train_1.shape[0],1)),np.ones((img_train_2.shape[0],1))))
label_train=label_train.ravel()
label_test=np.vstack((np.zeros((img_test_1.shape[0],1)),np.ones((img_test_2.shape[0],1))))
label_test=label_test.ravel()

del img_train_1_1D, img_train_1_2D,img_train_2_1D, img_train_2_2D, img_train_1,img_train_2,img_test_1,img_test_2
gc.collect()
time.sleep(0.5)
pass


# In[13]:


#### HOG-Gray: train
train_hogg=HOG_Gray(img_train) 
model1 =XGBClassifier(learning_rate=0.1, max_depth=5,n_estimators=100,silent=True, objective='binary:logistic')
print ("XGBoost -- training :" )
xgbt_model = model1.fit(train_hogg,label_train)  # xgbt training model
print ("training finished")


# In[14]:


#### HOG-Gray: test
print ("XGBoost -- predictining")
time_start=time.time()
test_hogg=HOG_Gray(img_test)
time_end=time.time()
print('HOG-Gray cost',time_end-time_start)

time_start=time.time()
xgbt_test= xgbt_model.predict(test_hogg)
time_end=time.time()
print('XGBoost cost',time_end-time_start)

accuracy_xgbt=(len(label_test)-sum(abs(xgbt_test-label_test)))/len(label_test)
print('XGBoost ACCURACY',accuracy_xgbt)
auc_xgbt = metrics.roc_auc_score(label_test,xgbt_test)
print('XGBoost AUC',auc_xgbt)

print ("**********************\n")

del train_hogg, test_hogg
gc.collect()
time.sleep(0.5)
pass


# In[15]:


os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
img_train = img_train/255
img_train.resize([img_train.shape[0], img_train.shape[1], img_train.shape[2], 1])
img_test_v = img_test/255
img_test_v.resize([img_test_v.shape[0], img_test_v.shape[1], img_test_v.shape[2], 1])
label_train_binary = to_categorical(label_train)
label_test_binary = to_categorical(label_test)


# In[17]:


###AlexNet

model_alex1 = Sequential()
input_shape = (IMG_SIZE,IMG_SIZE, 1)
model_alex1.add(Convolution2D(96, (11, 11), input_shape=input_shape,strides=(4, 4),  padding='valid',activation='relu', kernel_initializer='uniform'))
model_alex1.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model_alex1.add(Convolution2D(128, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
model_alex1.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model_alex1.add(Convolution2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
model_alex1.add(Convolution2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
model_alex1.add(Convolution2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
model_alex1.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model_alex1.add(Flatten())
# model.add(Dense(4096, activation='relu'))
model_alex1.add(Dense(1024, activation='relu'))
model_alex1.add(Dropout(0.5))
# model.add(Dense(4096, activation='relu'))
model_alex1.add(Dense(1024, activation='relu'))
model_alex1.add(Dropout(0.5))
model_alex1.add(Dense(2, activation='softmax'))
adam = keras.optimizers.Adam(lr=0.001)

#model_alex1 = multi_gpu_model(model_alex1,gpus=1)
model_alex1.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])
model_alex1.summary()


# In[18]:


###AlexNet training
History_alex1=model_alex1.fit(x=img_train,y=label_train_binary,
                       batch_size=BATCH_SIZE,
                       validation_data=(img_test_v,label_test_binary),
                       epochs=Epochs,
                       shuffle=True,
                       verbose=1)


# In[ ]:


#datagen = tf.keras.preprocessing.image.ImageDataGenerator()
#History_alex1=model_alex1.fit_generator(datagen.flow(img_train,label_train_binary,batch_size=BATCH_SIZE), 
#                             steps_per_epoch=len(label_train_binary) // BATCH_SIZE,  
#                             validation_data=datagen.flow(img_test_v,label_test_binary)
#                             validation_steps=len(label_test_binary) // BATCH_SIZE,
#                             epochs=Epochs)


# In[19]:



###AlexNet testing
print ("AlexNet -- predictining :")
time_start=time.time()
#img_test_v=img_test/255
#img_test_v.resize([img_test_v.shape[0], img_test_v.shape[1], img_test_v.shape[2], 1])
pred_alex1 = model_alex1.predict(img_test_v)
label_alex1=np.array(pred_alex1[:,0]<0.5)+0
time_end=time.time()
print('totally cost',time_end-time_start)

accuracy_alex1=(len(label_test)-sum(abs(label_alex1-label_test)))/len(label_test)
print(accuracy_alex1)
auc_alex1 = metrics.roc_auc_score(label_test,label_alex1)
print(auc_alex1)


# In[20]:



# plot the training loss and accuracy of AlexNet
N = Epochs
H = History_alex1
plt.style.use("ggplot")
plt.figure()
#plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
#plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_accuracy")
plt.title("accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("accuracy")
plt.legend(loc="lower right")
#plt.savefig(path_sav+"plot_acc_alex1.png")


# In[21]:


plt.style.use("ggplot")
plt.figure()
#plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
#plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.title("loss on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("loss")
plt.legend(loc="upper right")
#plt.savefig(path_sav+"plot_loss_alex1.png")
del model_alex1, History_alex1
gc.collect()
time.sleep(0.5)
pass


# In[5]:


try:   
    get_ipython().system('jupyter nbconvert --to python main.ipynb')
    # python即转化为.py，script即转化为.html
    # file_name.ipynb即当前module的文件名
except:
    pass

