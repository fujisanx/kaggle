# coding: utf-8

#------------------------------------------------------------
#    DNN model
#------------------------------------------------------------
import numpy as np
import pandas as pd
import gc

from keras.models import Model
from keras import applications
from keras import optimizers
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, AveragePooling2D, Conv2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping



# VGG16 image net acc is 71.5%, this network can work small images
# need larger than 32px
# custmized fuly connected layer for speedup
def VGG16(shape, num_classes, last_activation):
    base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=shape)
    x = base_model.output
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation=last_activation)(x)
    model = Model(base_model.input, x)
    
    return model


# image net acc is 75.2% , this network baseline, fast and good pefromance
# base size 224px, need  larger than 200px 
def ResNet50(shape, num_classes, last_activation):
    base_model = applications.ResNet50(weights='imagenet', include_top=False, input_shape=shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes, activation=last_activation)(x)
    model = Model(base_model.input, x)
    
    return model


# image net acc is 78.0%
# base image size 299px, need  larger than 150px 
def InceptionV3(shape, num_classes, last_activation):
    base_model = applications.InceptionV3(weights='imagenet', include_top=False, input_shape=shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes, activation=last_activation)(x)
    model = Model(base_model.input, x)
    
    return model


# image net acc is 79.1 , this network is best peroformance in keras pre-train model
# base image size 299px,  need larger than 150px 
def Xception(shape, num_classes, last_activation):
    base_model = applications.Xception(weights='imagenet', include_top=False, input_shape=shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes, activation=last_activation)(x)
    model = Model(base_model.input, x)
    
    return model


# image net acc is 78.0%
# base size 224px, need  larger than 200px 
import models.ResNet101 as ResNet101Model
def ResNet101(shape, num_classes, last_activation):
    base_model = ResNet101Model.ResNet( input_shape=shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes, activation=last_activation)(x)
    model = Model(base_model.input, x)
    
    return model


#  InceptionV1 (GoogLeNet) image net acc is 69.8, this network is quickly.
#  base size 224px, larger than 150px 
import models.InceptionV1 as InceptionV1Model
def InceptionV1(shape, num_classes, last_activation):
    base_model = InceptionV1Model.InceptionV1(include_top=False, input_shape=shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes, activation=last_activation)(x)
    model = Model(base_model.input, x)
    
    return model


# image net acc is 74.91 ,
# base image size 224px,  need larger than 112px 
# weigths download: 
import models.DenseNet121 as DenseNet121Model
def DenseNet121(shape, num_classes, last_activation):
    base_model = DenseNet121Model.DenseNet(weights='imagenet', input_shape=shape)
    x = base_model.output
    x = Dense(num_classes, activation=last_activation)(x)
    model = Model(inputs=base_model.input, outputs=x)
    return model

# image net acc is 77.64
# base image size 224px,  need larger than 112px 
import models.DenseNet161 as DenseNet161Model
def DenseNet161(shape, num_classes, last_activation):
    base_model = DenseNet161Model.DenseNet(weights='imagenet', input_shape=shape)
    x = base_model.output
    x = Dense(num_classes, activation=last_activation)(x)
    model = Model(inputs=base_model.input, outputs=x)
    return model


import models.squeezenet as SqueezeNetModel
def SqueezeNet(shape, num_classes, last_activation):
    base_model = SqueezeNetModel.get_squeezenet_top(input_shape=shape)
    x = base_model.layers[-1].output
#    x = Convolution2D(num_classes, 1, 1, border_mode='valid', name='conv10')(x)
#    x = Activation('relu', name='relu_conv10')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes, activation=last_activation)(x)
    model = Model(base_model.input, x)
    return model



def get_freeze_layer_num (freeze_type, network_name):
    freeze_layer = 0
    
    # if dataset class in imagenet, train speed is fast
    if freeze_type == 'finetune':
        if network_name == 'ResNet50':
            freeze_layer = 173

        elif network_name == 'VGG16':
            freeze_layer = 20

        elif network_name == 'InceptionV3':
            freeze_layer = 311

        elif network_name == 'Xception':
            freeze_layer = 132
            
        elif network_name == 'DenseNet161':
            freeze_layer = 806
            
        elif network_name == 'DenseNet121':
            freeze_layer = 806
            
    # freeze top conv block for train speed up, this is not affect performance 
    if freeze_type == 'topblock':
        if network_name == 'ResNet50' or network_name == 'ResNet101' :
            freeze_layer = 17

        elif network_name == 'VGG16':
            freeze_layer = 4

        elif network_name == 'InceptionV3':
            freeze_layer = 18

        elif network_name == 'Xception':
            freeze_layer = 16
            
        elif network_name == 'ResNet101':
            freeze_layer = 51

        elif network_name == 'DenseNet161':
            freeze_layer = 72
            
        elif network_name == 'DenseNet121':
            freeze_layer = 72
            
    print('freeze_layer=', freeze_layer)
            
    return freeze_layer



# tuned GTX1080ti 11GB memory
def get_bach_size(network_name, pixcel_length, freeze_layer_type):
    size = 24

    if freeze_layer_type == None:
        if network_name == 'InceptionV1':
            if pixcel_length == 224:
                size = 160
            elif pixcel_length == 256:
                size = 128
            elif pixcel_length == 299:
                size = 96
                
        elif network_name == 'VGG16' or network_name == 'ResNet50':
            if pixcel_length == 64:
                size = 512
            elif pixcel_length == 128:
                size = 256
            elif pixcel_length == 224:
                size = 72
            elif pixcel_length == 256:
                size = 56
            elif pixcel_length == 299:
                size = 42
                
        elif network_name == 'InceptionV3' or network_name == 'Xception':
            if pixcel_length == 224:
                size = 52
            elif pixcel_length == 256:
                size = 36
            elif pixcel_length == 299:
                size = 24
                
        elif network_name == 'DenseNet121':
            if pixcel_length == 224:
                size = 52
            elif pixcel_length == 256:
                size = 36
            elif pixcel_length == 299:
                size = 24
            elif pixcel_length == 128:
                size = 128
            elif pixcel_length == 64:
                size = 256
                
        elif network_name == 'DenseNet161':
            if pixcel_length == 224:
                size = 38
            elif pixcel_length == 256:
                size = 24
            elif pixcel_length == 299:
                size = 18
            elif pixcel_length == 128:
                size = 56
            elif pixcel_length == 64:
                size = 112
   
    
    # freeze top block
    if freeze_layer_type == 'topblock':
        
        if network_name == 'InceptionV1' or network_name == 'VGG16':
            if pixcel_length == 64:
                size = 512
            elif pixcel_length == 128:
                size = 320
            elif pixcel_length == 224:
                size = 172
            elif pixcel_length == 256:
                size = 132
            elif pixcel_length == 299:
                size = 96
            elif pixcel_length == 128:
                size = 320
                
        elif network_name == 'ResNet50':
            if pixcel_length == 224:
                size = 96
            elif pixcel_length == 256:
                size = 72
            elif pixcel_length == 299:
                size = 56
                
                
        elif network_name == 'InceptionV3' or network_name == 'Xception':
            if pixcel_length == 224:
                size = 64
            elif pixcel_length == 256:
                size = 52
            elif pixcel_length == 299:
                size = 38
                
        elif network_name == 'DenseNet121' or network_name == 'ResNet101':
            if pixcel_length == 224:
                size = 64
            elif pixcel_length == 256:
                size = 52
            elif pixcel_length == 299:
                size = 38
            elif pixcel_length == 128:
                size = 128

        elif network_name == 'ResNet101':
            if pixcel_length == 224:
                size = 38
            elif pixcel_length == 256:
                size = 32
            elif pixcel_length == 299:
                size = 24
                
        elif network_name == 'DenseNet161':
            if pixcel_length == 224:
                size = 24
            elif pixcel_length == 256:
                size = 18
            elif pixcel_length == 299:
                size = 12
            elif pixcel_length == 128:
                size = 64
    
    # finetune can large bach size
    if freeze_layer_type == 'finetune':
        if network_name == 'InceptionV1' or network_name == 'VGG16' or network_name == 'ResNet50':
            size = 384
        elif network_name == 'InceptionV3' or network_name == 'Xception':
            size = 256
            
    print('bach size=', size)
    return size



# tuned for pretrain model training
def get_optimizer(optimizer_type,freeze_layer_type):
    if optimizer_type == 'Adam':
        opt = optimizers.Adam(lr=1e-5)
    elif optimizer_type == 'SGD':
        opt = optimizers.SGD(lr=5e-4, momentum=0.9)
    return opt


#------------------------------------------------------------
#    for dataset
#------------------------------------------------------------
#split dataset train and test 8:2
def split_dataset_holdout(x, y, split_num):
    val_split_num = int(round(split_num*len(y)))
    x_train = x[:val_split_num]
    y_train = y[:val_split_num]
    x_test = x[val_split_num:]
    y_test = y[val_split_num:]
    return x_train, y_train, x_test, y_test


# split dataset train and test , get 
# return data ype add: use large dataset avoid memory error 
def split_dataset_leave_one_out(x, y, return_data_type, split_num, test_data_no):
    data_num = len(y)
    
    test_start = int( (data_num / split_num) * test_data_no)
    test_end = int(((data_num / split_num) * test_data_no) + (data_num / split_num) )
    
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    test_str = []
    for i in range( len(y) ):
        if i >= test_start and i < test_end:
            x_test.append(x[i])
            y_test.append(y[i])
            test_str.append(i)
        else:
            x_train.append(x[i])
            y_train.append(y[i])
            
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    
    if return_data_type == 'train':
        return x_train, y_train
    else:
        return x_test, y_test


# shuffle dataset 
def shuffle_dataset(x, y):
    data_num = len(y)
    random_index = np.random.permutation(data_num)

    x_shuffle = []
    y_shuffle = []
    no_shuffle = []
    for i in range(data_num):
        x_shuffle.append(x[random_index[i]])
        y_shuffle.append(y[random_index[i]])
        no_shuffle.append(i)
    x = np.array(x_shuffle) 
    y = np.array(y_shuffle)
    no = np.array(no_shuffle)
    
    return x, y, no


# if dataset unbalance
from collections import Counter
def dataset_rebalance_under_sampling(x, y, target_data, max_num):
    counter = Counter(target_data)
    max_balance_num = max_num
    
    xx = []
    yy = []
    unique_ids = []
    for i, _y in enumerate(target_data):
        random_num = np.random.rand()
        add_fl = False

        if counter[_y] >= max_balance_num:
            if random_num <= (max_balance_num / counter[_y]) :
                add_fl = True
        else:
            add_fl = True

        if add_fl == True:
            xx.append(x[i])
            yy.append(y[i])
            unique_ids.append(_y)

    xx = np.array(xx)
    yy = np.array(yy)
    unique_ids = np.array(unique_ids)
    
    counter = Counter(unique_ids)
    #plt.hist(unique_ids, bins=441)
    #plt.show()
    
    return xx, yy

# flip & rotate image for test time augmentation
def tta_images(images, flip='normal', rotate='normal'):
    if flip=='lr':
        for i, image in enumerate(images):
            images[i] = np.fliplr(image)
        
    elif flip=='ud':
        for i, image in enumerate(images):
            images[i] = np.flipud(image)
        
    elif flip=='udlr':
        for i, image in enumerate(images):
            images[i] = np.flipud( np.fliplr(image) )

    if rotate==90:
        for i, image in enumerate(images):
            images[i] = np.rot90(image)

    images = images.astype('float32')
    images /= 255
    
    print('loaded.. flip=', flip, ' rotate=', rotate)
    return images


#------------------------------------------------------------
#    for ensemble
#------------------------------------------------------------
def mean_ensemble(predictions):
    num_classes=len(predictions[0][0])
    r = []
    for i in range( len(predictions[0]) ):
        p = np.zeros(num_classes)
        for j in range( len(predictions) ):
            p += predictions[j][i]
        p /= len(predictions)
        r.append(p)
    
    return r


#------------------------------------------------------------
#    for visualize
#------------------------------------------------------------

import matplotlib.pyplot as plt
# Keras loss acc history
def plot_history(history):
    print(history.history.keys())
    plt.subplot(121)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.subplot(122)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    