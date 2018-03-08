import tensorflow as tf
import numpy as np

def dense(input, nodes, regularizer, dropout_rate, training) : 
    x = tf.layers.dropout(input, dropout_rate, training=training)
    x = tf.layers.dense(x, nodes)
    #x = tf.layers.batch_normalization(x, training=training)
    x = tf.nn.elu(x)
    return x

def convolution(input, filters, regularizer, dropout_rate, training) : 
    x = tf.layers.dropout(input, dropout_rate, training=training)
    x = tf.layers.conv2d(x, filters, 3, 1, padding='same', kernel_regularizer=regularizer, bias_regularizer=regularizer)
    #x = tf.layers.batch_normalization(x, training=training)
    x = tf.nn.elu(x)
    return x

def deconvolution(input, filters, regularizer, dropout_rate, training) : 
    x = tf.layers.dropout(input, dropout_rate, training=training)
    x = tf.layers.conv2d_transpose(x, filters, 3, 1, padding='same', kernel_regularizer=regularizer, bias_regularizer=regularizer)
    #x = tf.layers.batch_normalization(x, training=training)
    x = tf.nn.elu(x)
    return x

def encoder(filters, code_size, regularizer, dropout_rate):
    """
    Args:
        - code_size: number of nodes in code layer
        - filters: number of filters per convolution
        - regularizer: the regularizer to use
        - dropout_rate: chance of dropping a node
    """
    input = tf.placeholder(tf.float32, [None, 32, 32, 3], name='encoder_input')
    training = tf.placeholder_with_default(False, [], name='encoder_training')
    
    with tf.variable_scope('encoder'):
        x = input
        for i in range(len(filters)):
            if filters[i]==0:
                x = tf.layers.max_pooling2d(x, 2, 2, padding='same')
            else:
                x = convolution(x, filters[i], regularizer, dropout_rate, training)
        precode_shape = x.get_shape().as_list()
        x = tf.reshape(x, [-1, np.prod(precode_shape[1:])])
        x = dense(x, code_size, regularizer, dropout_rate, training)
        
    output = tf.identity(x, name='encoder_output')
    return (input, output, training, precode_shape)

def decoder(encoder_output, postcode_shape, filters, regularizer, dropout_rate):
    """
    Args:
        - encoder_output: output tensor from decoder
        - postcode_shape: shape to reshape tensor to after code layer
        - filters: number of filters per convolution
        - regularizer: the regularizer to use
        - dropout_rate: chance of dropping a node
    """
    input = tf.identity(encoder_output, name='decoder_input')
    training = tf.placeholder_with_default(False, [], name='decoder_training')
    
    with tf.variable_scope('decoder'):
        x = input
        x = dense(x, np.prod(postcode_shape[1:]), regularizer, dropout_rate, training)
        x = tf.reshape(x, [-1, postcode_shape[1], postcode_shape[2], postcode_shape[3]])
        for i in reversed(range(len(filters))):
            if filters[i]==0:
                x = tf.image.resize_nearest_neighbor(x, [x.get_shape().as_list()[1]*2, x.get_shape().as_list()[2]*2])
            else:
                x = deconvolution(x, filters[i], regularizer, dropout_rate, training)
        x = deconvolution(x, 3, regularizer, dropout_rate, training)
        
    output = tf.identity(x, name='decoder_output')
    return (input, output, training)

def psnr(original, reconstruction):
    """
    Args:
        - original: 4D tensor of shape NHWC, the original image
        - reconstruction: 4D tensor of shape NHWC, the reconstructed image
    """
    with tf.variable_scope('psnr') as scope:
        x = tf.losses.mean_squared_error(original, reconstruction, reduction = tf.losses.Reduction.NONE)
        x = tf.reduce_mean(x, axis=3)
        x = tf.reduce_mean(x, axis=2)
        x = tf.reduce_mean(x, axis=1)
        x = 20.0 * tf.log(255.0) / tf.log(10.0) - 10.0 * tf.log(x) / tf.log(10.0)
    return x

# Based on "Densely Connected Convolutional Networks" by Huang et al.
def dense_block(inputs, layers, k, bottleneck, dropout_rate, is_training):
    """
    Args:
        - inputs: 4D tensor of shape NHWC
        - layers: number of convolutional layers
        - k: number of filters per convolution
        - bottleneck: whether to number of reduce input feature maps
        - dropout_rate: chance of dropping a node
        - is_training: boolean scalar tensor
    """
    with tf.variable_scope('dense_block') as scope:
        concatenation = inputs
        for i in range(layers):
            x = concatenation
            if bottleneck:
                x = tf.layers.batch_normalization(x, training=is_training)
                x = tf.nn.elu(x)
                x = tf.layers.conv2d(x, 4*k, 1, 1, padding='same')
            x = tf.layers.batch_normalization(x, training=is_training)
            x = tf.nn.elu(x)
            x = tf.layers.conv2d(x, k, 3, 1, padding='same')
            x = tf.layers.dropout(x, dropout_rate, training=is_training)
            concatenation = tf.concat([concatenation, x], 3)
    return concatenation

# Based on "Densely Connected Convolutional Networks" by Huang et al.
def transition_layer(inputs, compression, dropout_rate, is_training):
    """
    Args:
        - inputs: 4D tensor of shape NHWC
        - compression: factor to reduce number of feature maps
        - dropout_rate: chance of dropping a node
        - is_training: boolean scalar tensor
    """
    with tf.variable_scope('transition_layer') as scope:
        x = tf.layers.batch_normalization(inputs, training=is_training)
        numFilters = tf.floor(tf.shape(inputs)[3]*compression)
        x = tf.layers.conv2d(x, numFilters, 1, 1, padding='same')
        x = tf.layers.average_pooling2d(x, 2, 2, padding='same')
    return x
