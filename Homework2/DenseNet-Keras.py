from keras import regularizers
from keras import callbacks
from keras.models import Sequential, Model, load_model
from keras.utils import to_categorical, plot_model
from keras.layers import Input, Dense, Reshape, Lambda, LeakyReLU, BatchNormalization, Concatenate
from keras.layers.core import Activation, Flatten, Dropout
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import SGD, Adam, RMSprop
from keras.datasets import mnist, fashion_mnist
import keras.backend as K
from keras.regularizers import l2
from keras.utils.vis_utils import model_to_dot
import numpy as np
import scipy
import tensorflow as tf
import argparse
import math
import os
import cv2
import time

def conv_factory(x, concat_axis, nb_filter,
                 dropout_rate=None, weight_decay=1E-4):
    """
    Apply BatchNorm, Relu 3x3Conv2D, optional dropout

    Inputs:
    =======
    x: Input keras network
    concat_axis: int -- index of contatenate axis
    nb_filter: int -- number of filters
    dropout_rate: float -- dropout rate
    weight_decay: float -- weight decay factor

    Outputs:
    ========
    x: keras model -- network with b_norm, relu and Conv2D added
    """

    x = BatchNormalization(axis=concat_axis,
                           gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = Activation('elu')(x)
    x = Conv2D(nb_filter, (3, 3),
               kernel_initializer="he_uniform",
               padding="same",
               kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x

def transition_down(x, concat_axis, nb_filter,
                    dropout_rate=None, weight_decay=1E-4):
    """
    Apply BatchNorm, Relu, 1x1 Conv2D, optional dropout and Maxpooling2D
    
    Inputs:
    =======
    x: keras model
    concat_axis: int -- index of contatenate axis
    nb_filter: int -- number of filters
    dropout_rate: float -- dropout rate
    weight_decay: float -- weight decay factor
    
    
    Outputs:
    ========
    x: keras model
    """

    x = BatchNormalization(axis=concat_axis,
                           gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = Activation('elu')(x)
    x = Conv2D(nb_filter, (1, 1),
               kernel_initializer="he_uniform",
               padding="same",
               kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    x = AveragePooling2D((2, 2), strides=(2, 2))(x)

    return x

def denseblock_down(x, concat_axis, nb_layers, nb_filter, growth_rate, 
                    dropout_rate=None, weight_decay=1E-4):
    """
    Build a denseblock where the output of each conv_factory is fed to subsequent ones
       
    Inputs:
    =======
    x: keras model
    concat_axis: int -- index of contatenate axis
    nb_layers: int -- the number of layers of conv_factory to append to the model
    nb_filter: int -- used to keep track of total number of filters
    dropout_rate: int -- dropout rate
    weight_decay: int -- weight decay factor
    
    Outputs: 
    ========
    x: keras model
    nb_filter: int -- used to keep track of total number of filters
    """

    list_feat = [x]

    for i in range(nb_layers):
        x = conv_factory(x, concat_axis, growth_rate,
                         dropout_rate, weight_decay)
        list_feat.append(x)
        x = Concatenate(axis=concat_axis)(list_feat)
        nb_filter += growth_rate

    return x, nb_filter

def DenseNet(nb_classes, img_dim, convblock_per_DB=4, DB_per_stage=2, 
             growth_rate=16, nb_filter=32, upsampling_mode='upsampling2d', 
             dropout_rate=0.2, weight_decay=1E-4):
    """ Build the DenseNet model
    :param nb_classes: int -- number of classes
    :param img_dim: tuple -- (channels, rows, columns)
    :param depth: int -- how many layers
    :param nb_dense_block: int -- number of dense blocks to add to end
    :param growth_rate: int -- number of filters to add
    :param nb_filter: int -- number of filters
    :param dropout_rate: float -- dropout rate
    :param weight_decay: float -- weight decay
    :returns: keras model with nb_layers of conv_factory appended
    :rtype: keras model
    """
    
    if K.image_dim_ordering() == "th":
        concat_axis = 1
    elif K.image_dim_ordering() == "tf":
        concat_axis = -1

    model_input = Input(shape=img_dim)

    # Initial Conv layer =========================================
    x = Conv2D(nb_filter, (3, 3),
               kernel_initializer="he_uniform",
               padding="same",
               name="initial_conv2D",
               kernel_regularizer=l2(weight_decay))(model_input)

    # Downsampling (DenseBlock - TransitionDown) * N =============
    concat_node_list = []
    for block_idx in range(DB_per_stage):
        concat_feature_list = []
        concat_feature_list.append(x)
        x, nb_filter = denseblock_down(x, concat_axis, convblock_per_DB,
                                       nb_filter, growth_rate, 
                                       dropout_rate=dropout_rate,
                                       weight_decay=weight_decay)
        concat_feature_list.append(x)
        concat_node = Concatenate(concat_axis)(concat_feature_list)
        concat_node_list.append(concat_node)
        x = transition_down(x, concat_axis, nb_filter, dropout_rate=dropout_rate,
                            weight_decay=weight_decay)

    x = BatchNormalization(axis=concat_axis,
                           gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Flatten()(x)
    x = Dense(nb_classes,
              activation='softmax',
              kernel_regularizer=l2(weight_decay),
              bias_regularizer=l2(weight_decay))(x)

    densenet = Model(inputs=[model_input], outputs=[x], name="DenseNet")

    return densenet

learning_rate = 0.001
callback_function = callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=5, verbose=0, mode='auto')
callback_list = [callback_function]

for num_DB in range(1, 4):
    for conv_per_DB in range(1, 6):
        fold_array = []
        for fold in range(4):
            net = DenseNet(nb_classes = 7, 
                           img_dim = (129, 129, 1), 
                           convblock_per_DB = conv_per_DB,
                           DB_per_stage = num_DB,
                           growth_rate = 2, 
                           nb_filter = 2)
            learning_rate = 0.001
            net.compile(Adam(lr = learning_rate, decay = 1e-2), 
                        loss = 'categorical_crossentropy', 
                        metrics=['accuracy'])
            # load training data
            train_images = np.load(data_dir + 'train_x_' + str(fold+1) + '.npy', mmap_mode='r+')
            train_labels = np.load(data_dir + 'train_y_' + str(fold+1) + '.npy')
            train_images = np.reshape(train_images, (-1, 129, 129, 1))
            # load testing data
            test_images = np.load(data_dir + 'test_x_' + str(fold+1) + '.npy', mmap_mode='r+')
            test_labels = np.load(data_dir + 'test_y_' + str(fold+1) + '.npy')
            test_images = np.reshape(test_images, (-1, 129, 129, 1))
            history = net.fit(x = train_images, 
                              y = train_labels, 
                              batch_size = 32, 
                              epochs = 200, 
                              validation_data = (test_images, test_labels), 
                              verbose = 0, 
                              callbacks = callback_list)
            acc = net.evaluate(x = test_images, y = test_labels)[1]
            print('accuracy: ', acc)
            fold_array = np.append(fold_array, acc)
            plt.figure()
            plt.plot(history.history['acc'])
            plt.plot(history.history['val_acc'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()
        print('Number of DB: ', num_DB)
        print('Conv per DB: ', conv_per_DB)
        print('Total filters: ', net.layers[-3].get_output_shape_at(0)[-1])
        print('Acc array across folds: ', fold_array)
        print('==============================================')