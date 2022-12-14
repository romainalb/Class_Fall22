# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 15:47:35 2022

@author: albou
"""
import os
import tensorflow as tf
from random import seed
from random import randint
from PIL import Image
import numpy as np
import scipy.io as sio
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import time
seed(1) 
#gpu = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(gpu[0], True)
#os.environ["XLA_FLAGS"]="--C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12"


nimagestrain= 1 #Number of images used for training
nimagestest = 1
npixels_train = 7500 #Number of pixels per images used for training
npixels_test =1000
ntrain = nimagestrain*npixels_train   # per class. 300000 for full training, maybe test lower values for testing that the algorithm can overfit an image first
ntest = nimagestest*npixels_test # Will depend on the images I make for testing 
nclass = 2  #  number of classes
imsize = 33
nchannels = 3


Train = np.zeros((ntrain*nclass,imsize,imsize,nchannels))
Test = np.zeros((ntest*nclass,imsize,imsize,nchannels))
LTrain = np.zeros((ntrain*nclass,nclass))
LTest = np.zeros((ntest*nclass,nclass))


os.chdir('D:/Research/Class_Fall22/FInal_Project/Code/960G/Code/1') #training folder
test = sio.loadmat('Labels.mat')
loaded = test['num_image']

itrain = -1

'''for iimages in range (0,nimagestrain):
    for isample in range (0,ntrain*nclass):
        im = plt.imread('Image%d.png'%(isample+1)) 
        itrain += 1
        Train[itrain,:,:,0] = im[:,:,0] #Figure out how to do the bicubic
        Train[itrain,:,:,1] = im[:,:,1]
        Train[itrain,:,:,2] = im[:,:,2] 
        LTrain[itrain,:] = loaded[itrain,:] # 1-hot lable
 '''       
        
        
for iimages in range (0,nimagestrain):
    for isample in range (0,npixels_train*nclass):
        im = plt.imread('Image%d.png'%(isample+1)) 
        itrain += 1
        Train[itrain,:,:,0] = im[:,:,0] #Figure out how to do the bicubic
        Train[itrain,:,:,1] = im[:,:,1]
        Train[itrain,:,:,2] = im[:,:,2] 
        LTrain[itrain,:] = loaded[itrain,:] # 1-hot lable
        
os.chdir('D:/Research/Class_Fall22/FInal_Project/Code/960G/Code/1_test') #testing folder
test = sio.loadmat('Labels.mat')
loaded = test['num_image2']

itest = -1
for iimages in range (0,nimagestest):
    for isample in range (0,npixels_test*nclass):
        im = plt.imread('Image%d.png'%(isample+1))
        itest += 1
        Test[itest,:,:,0] = im[:,:,0] 
        Test[itest,:,:,1] = im[:,:,1] 
        Test[itest,:,:,2] = im[:,:,2]  
        LTest[itest,:] = loaded[itest,:]

#with tf.device('/CPU:0'):
model = models.Sequential()  
   # normalization_layer = layers.Rescaling(1./255) #standardize the data
model.add(layers.Conv2D(30, (5, 5), groups = 3, 
                        bias_initializer = tf.keras.initializers.Ones(),
                        kernel_initializer = tf.keras.initializers.GlorotNormal(), 
                        activation='relu', 
                        input_shape=(33, 33, 3)))
model.add(layers.MaxPooling2D((2, 2),strides=(2, 2)))
model.add(tf.keras.layers.Reshape((14, 14, 10, 3)))
model.add(layers.Conv3D(45, (5, 5, 10), groups = 3,
                        bias_initializer = tf.keras.initializers.Ones(),
                        kernel_initializer = tf.keras.initializers.GlorotNormal(), 
                        activation='relu',
                        input_shape=(14, 14, 10, 3)))
model.add(tf.keras.layers.Reshape((10, 10, 45)))
model.add(layers.MaxPooling2D((2, 2),strides=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(100, activation='relu',
                       bias_initializer = tf.keras.initializers.Ones(), #No regularization on the weights according to eq (2)
                       kernel_initializer = tf.keras.initializers.GlorotNormal(),
                       kernel_regularizer = tf.keras.regularizers.L2(l2=0.1))) #model says it uses regularization, will try on only last layer for now
model.add(layers.Dense(2, activation='softmax',
                       bias_initializer = tf.keras.initializers.RandomNormal()))

model.summary()

model.compile(optimizer=tf.keras.optimizers.experimental.Adam(learning_rate=0.001),   #Following the paper I am using 
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),  #False because softmax is used on the last layer
              metrics=['accuracy'])

history = model.fit(Train, LTrain,verbose=2, batch_size=10, epochs=5, validation_data=(Test, LTest))
start_time = time.time()
print("--- %s seconds ---" % (time.time() - start_time))