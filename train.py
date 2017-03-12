# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 22:44:24 2017

@author: vijay
"""

import numpy as np
import cv2
import os
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

from keras.preprocessing.image import ImageDataGenerator

###############################################################################

def cnn_model():

    model = Sequential()
 
     ## Conv layers
 
    model.add(Convolution2D(128, 3, 3, border_mode='same', input_shape=(3, 128, 128), activation='relu'))
 
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
 
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
     ## Dropout
 
    model.add(Dropout(0.2))
    model.add(Flatten())
 
     ## Hidden layers 
 
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    # Compile model
    adamax=keras.optimizers.Adagrad(lr=0.001, epsilon=1e-08, decay=0.0)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=adamax, metrics=['accuracy'])
 
    return model

###############################################################################

def img_mapping(loc):
    
    imgs=[]
    labels=[]
    
    classes=os.listdir(loc)
    
    for clas in classes:
        
        imlist=os.listdir(loc+clas)
        
        for im in imlist:
            
            imgs.append(loc+clas+'/'+im)
            labels.append(clas)
            
            
    return (imgs,labels)
    
###############################################################################
    
def minibatch_generator(imgs,labels,batchsize):
    
    images=[]    
    
    l=len(imgs)
    rand=np.random.uniform(0,l,batchsize).astype(np.int).tolist()
    
    img_batch=[imgs[i] for i in rand]
    labels_batch=[labels[i] for i in rand]
    
    for indx,img in enumerate(img_batch):
        
        i=cv2.imread(img).astype('float32')
        i=i.reshape((1,3,i.shape[0],i.shape[1]))
        i=i/255
        images.append(i)
        
    return (images,labels_batch)   
    
###############################################################################
    
def augment(imgs,labels,batchsize):

    datagen = ImageDataGenerator(featurewise_center=False, # set input mean to 0 over the dataset
                                 samplewise_center=False, # set each sample mean to 0
                                 featurewise_std_normalization=True, # divide inputs by std of the dataset
                                 samplewise_std_normalization=False, # divide each input by its std
                                 zca_whitening=False, # apply ZCA whitening
                                 rotation_range=20, # randomly rotate images in the range (degrees, 0 to 180)
                                 width_shift_range=0.2, # randomly shift images horizontally (fraction of total width)
                                 height_shift_range=0.2, # randomly shift images vertically (fraction of total height)
                                 horizontal_flip=False, # randomly flip images
                                 vertical_flip=False) # randomly flip images
                                 
#    (sample,labels)=minibatch_generator(imgs,labels,batchsize)
#    
#    for X_sample in sample:
#        datagen.fit(X_sample.reshape((1,3,X_sample.shape[0],X_sample.shape[1])).astype(np.uint8)) # let's say X_sample is a small-ish but statistically representative sample of your data
#        
#    print 'Image augmenter is ready !'
    
    return datagen

###############################################################################

def main():
    
    train_root='/media/vijay/Vijay/Vijay/Github/keras-digits/dataset/train/'
    
    (imgs,labels)=img_mapping(train_root)
    
    batchsize=16
    
    epochs=10
    mb_per_epoch=len(imgs)/batchsize
    
#    datagen=augment(imgs,labels,1024)
    
    ## start training
    
    model=cnn_model()

    for epoch in range(epochs):
        
        print 'Entering epoch no ',epoch
        
        for mb_number in range(mb_per_epoch):
            
            print 'Entering minibatch no ',mb_number, 'out of ',mb_per_epoch
            
            (mb_train,mb_labels)=minibatch_generator(imgs,labels,batchsize)
            mb_labels=np.array(mb_labels)
            model.fit(np.vstack(mb_train),mb_labels, batch_size=batchsize, nb_epoch=1)
                        
        model.save_weights('digits_'+str(epoch)+'.h5')
    


if __name__=='__main__':
    main()