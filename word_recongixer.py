# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 22:14:59 2018

@author: Efas
"""


import keras


from keras.models import Sequential
from keras.layers.core import Dense, Dropout,Activation,Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
#from keras.layers import GlobalAveragePooling2D,GlobalAveragePooling1D, Lambda
#from keras.optimizers import SGD,RMSprop,adam
#from keras.utils import np_utils

import numpy as np

import matplotlib 
import os #makes more portable and gives access to the system
from PIL import Image
from numpy import *

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import cv2

#path1 =r"G:\pics\before_reform_originalS"
path1=r"G:\pics\pilo\before_reform_originals\0"#put the folder containing the image preprocessed after running pilo.py
path2 =r"G:\pics\after_reform" #put folder where you wish to store the new images 
#path2 =r"G:\pics\pilo\before_reform_originals\0" 
cvpath=path2



image_size=64#56
filter_number=32#64
Batch_size=10#64
dropoutvar=0.2# default for so many days 0.3
classsize=100
classwidth=250
#image_size=32#110
cv_imsize=(image_size,image_size)


listening = os.listdir(path1)
num_samples = size(listening)

#for file in listening:
 #   im = Image.open(path1+'\\'+file)
  #  img = im.resize((image_size,image_size))
   # gray = img.convert('L')
   # gray.save(path2+'\\'+file,"JPEG")
   
   
   
   
   
for file in listening:
    #im = Image.open(path1+'\\'+file)
    im = cv2.imread(path1+'\\'+file)
    
    #img = im.resize((image_size,image_size))
    #gray = img.convert('L')
    gray= cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    (thresh,bn)=cv2.threshold(gray,128,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    #bn.save(path2+'\\'+file,"JPEG")
    img=cv2.resize(bn,cv_imsize,interpolation = cv2.INTER_AREA)
    cv2.imwrite(cvpath+'\\'+file,img) 
    
    
    
    
    
############################################################    
 ########################################################   
    
    
    

    
    
###################################################### #   
##################################################    
    
    
    
    
    
    

im_list = os.listdir(path2)
num_samples = size(im_list)

iml = array([array(Image.open(path2+'\\'+im2)).flatten() for im2 in im_list], 'f')# store the flattened version of all the image in an array

label = np.ones((num_samples),dtype=int)

j=0;

_range = classsize*classwidth-1
for i in range(0,_range,classwidth):
    label[i:i+classwidth]=j
    j=j+1



im_matrix = iml
data,Label = shuffle(im_matrix,label, random_state = 2)
train_data = [data,Label]



(X,Y) = (train_data[0],train_data[1])
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.01,random_state = 4)

x_train= X_train.reshape(X_train.shape[0],image_size,image_size,1)
x_test=X_test.reshape(X_test.shape[0],image_size,image_size,1)

x_train/=255
x_test/=255

xjust= x_train

y_train= keras.utils.to_categorical(Y_train)
y_test= keras.utils.to_categorical(Y_test)


y_test= np.array(y_test)
y_train= np.array(y_train)

x_test= np.array(x_test)
x_train= np.array(x_train)

#image augmentation, if augmaentation is required

#datagen = keras.preprocessing.image.ImageDataGenerator(
#        featurewise_center=False,
#        samplewise_center=False,
#        featurewise_std_normalization=False,
#        samplewise_std_normalization=False,
#        zca_whitening=False,
#        rotation_range=45,
#        width_shift_range=0.2,
#        height_shift_range=0.2,
#        horizontal_flip=True,
#        vertical_flip=False
#        )
#
#datagen.fit(x_train)







#almost done with preprocessing
model = Sequential()
model.add(Convolution2D(filter_number,3,data_format='channels_last',activation='relu',input_shape = (image_size,image_size,1)))
# arguments explanation 32 = filter number, 3 = height and weightg of the convolution window, as far as I understand, it is the size of the filters
#there is an optional argument strider(1,1), and another called padding ="valid" or "same"
#data_fromat channels_last is for (batch, height, weight, channel) similar to my data fromat, channels_first would have the channel before height
#activision defines the activision function, relu insures that normalization layers dont need to be added saperately
model.add(MaxPooling2D(pool_size=(2,2)))# size of the pooling layers
model.add(Convolution2D(filter_number,3,data_format='channels_last',activation='relu',input_shape = (image_size,image_size,1)))
model.add(MaxPooling2D(pool_size=(2,2)))

##get rid of it if accuracy drops
model.add(Dropout(dropoutvar))
model.add(Convolution2D(filter_number,3,data_format='channels_last',activation='relu',input_shape = (image_size,image_size,1)))
model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Flatten())#flattens the model eg, flatten(3,4,5) =  60
model.add(Dense(100))# 100 nodes in the 1st hidden layer
model.add(Dropout(dropoutvar))# avoidsd overfitting
model.add(Dense(classsize))
model.add(Activation("softmax"))#optional, needed to incerease the accuricy
model.compile(loss="categorical_crossentropy", optimizer="adadelta", metrics = ["accuracy"]) #loss calculates the errors, optimizer learns the weights, metrices defines what the metrices are based on 

model.summary()
 

hystory1=model.fit(x_train,y_train,epochs=200,validation_data=(x_test,y_test)) 


pred=model.predict(x_test[0:10])



for i in range(10):
    print(pred[i])
    print(y_test[i])
    
model.summary()





model.save("3layermodel.h5")







#model = load_model("3layermodel.h5")
#model.summary()

import matplotlib.pyplot as plt
from keras.models import load_model
from keras.callbacks import History 
history = History()

plt.plot(hystory1.history['acc'])
plt.plot(hystory1.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss

plt.plot(hystory1.history['loss'])
plt.plot(hystory1.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


import pandas as pd
import seaborn as sn

df_cm = pd.DataFrame(pred, range(10),range(100))
sn.heatmap(df_cm, annot=True)







