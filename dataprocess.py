# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 20:07:25 2018

@author: 401-zhangjunnan
"""
import keras
from keras.callbacks import EarlyStopping,ModelCheckpoint
import numpy as np
import tensorflow as tf
from keras import models,layers,optimizers
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


data1 = np.loadtxt("3.0.txt");
data2 = np.loadtxt("3.1.txt");

val_data1 =np.loadtxt("4.0.txt");
val_data2 =np.loadtxt("4.1.txt");

[sample1,dimension] = data1.shape;
[sample2,dimension] = data2.shape;
[sample3,dimension] = val_data1.shape;
[sample4,dimension] = val_data2.shape;

label1 = np.ones([sample1,1]);
val_labels1 = np.ones([sample3,1]);

label2 = np.zeros([sample2,1]);
val_labels2 = np.zeros([sample4,1]);

newsample1 = np.concatenate((data1,label1),axis=1);
newsample2 = np.concatenate((data2,label2),axis=1);
val_sample1 = np.concatenate((val_data1,val_labels1),axis=1);
val_sample2 = np.concatenate((val_data2,val_labels2),axis=1);

datasample = np.concatenate((newsample1,newsample2),axis=0);
val_datasample = np.concatenate((val_sample1,val_sample2),axis=0);

Y = datasample[:,3];
Y_val = val_datasample[:,3];

X = datasample[:,0:3];
X_val = val_datasample[:,0:3];

model = models.Sequential();
model.add(layers.Dense(3,activation='relu',input_shape=(3,)));
model.add(layers.Dense(32,activation='relu'));
model.add(layers.Dense(10,activation='relu'));
model.add(layers.Dense(1,activation='sigmoid'));
model.compile(optimizer=optimizers.RMSprop(lr=0.01),loss='binary_crossentropy',metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='acc', patience=100)
checkpointer = ModelCheckpoint('resnet50_best-v2.h5', verbose=1, save_best_only=True)

log = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)

history = model.fit(X,Y,epochs=1000,callbacks=[log,early_stopping,checkpointer],validation_data=(X_val,Y_val))

model.save('densenet_final-v2.h5')