import tensorflow as tf
import numpy as np
import math

def create_model(input, nodes):
    """
    Generates a block of tensors
    
    Args:
        - input: the input tensor
    """
    with tf.name_scope('linear_model') as scope:
        hidden = tf.layers.Dense(nodes,
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.),
                             bias_regularizer=tf.contrib.layers.l2_regularizer(scale=1.),
                             activation=tf.nn.relu)
        output = tf.layers.Dense(10,
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.),
                             bias_regularizer=tf.contrib.layers.l2_regularizer(scale=1.))
    
        return output(hidden(input))