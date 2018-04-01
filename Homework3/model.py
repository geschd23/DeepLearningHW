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

def encoder(filters, code_size, regularizer, dropout_rate, normalize, data_type):
    """
    Args:
        - code_size: number of nodes in code layer
        - filters: number of filters per convolution
        - regularizer: the regularizer to use
        - dropout_rate: chance of dropping a node
        - normalize: whether to normalize input (/255)
        - data_type: data type to use on code layer
    """
    input = tf.placeholder(tf.float32, [None, 32, 32, 3], name='encoder_input')
    training = tf.placeholder_with_default(False, [], name='encoder_training')
    
    with tf.variable_scope('encoder'):
        x = input
        if normalize:
            x = x / 255
        for i in range(len(filters)):
            if filters[i]==0:
                x = tf.layers.max_pooling2d(x, 2, 2, padding='same')
            else:
                x = convolution(x, filters[i], regularizer, dropout_rate, training)
        precode_shape = x.get_shape().as_list()
        x = tf.reshape(x, [-1, np.prod(precode_shape[1:])])
        if code_size != 0:
            x = dense(x, code_size, regularizer, dropout_rate, training)
        x = tf.cast(x, data_type)
        
    output = tf.identity(x, name='encoder_output')
    return (input, output, training, precode_shape)

def decoder(encoder_output, code_size, postcode_shape, filters, regularizer, dropout_rate, normalize):
    """
    Args:
        - encoder_output: output tensor from decoder
        - code_size: number of nodes in code layer
        - postcode_shape: shape to reshape tensor to after code layer
        - filters: number of filters per convolution
        - regularizer: the regularizer to use
        - dropout_rate: chance of dropping a node
        - normalize: whether to (de)normalize output (*255)
        - data_type: data type to use on code layer
    """
    input = tf.identity(encoder_output, name='decoder_input')
    training = tf.placeholder_with_default(False, [], name='decoder_training')
    
    with tf.variable_scope('decoder'):
        x = tf.cast(input, tf.float32)
        if code_size != 0:
            x = dense(x, np.prod(postcode_shape[1:]), regularizer, dropout_rate, training)
        x = tf.reshape(x, [-1, postcode_shape[1], postcode_shape[2], postcode_shape[3]])
        for i in reversed(range(len(filters))):
            if filters[i]==0:
                x = tf.image.resize_nearest_neighbor(x, [x.get_shape().as_list()[1]*2, x.get_shape().as_list()[2]*2])
            else:
                x = deconvolution(x, filters[i], regularizer, dropout_rate, training)
        x = deconvolution(x, 3, regularizer, dropout_rate, training)
        if normalize:
            x = x * 255
            x = tf.maximum(tf.minimum(x, 255), 0)
        
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
        x = 20.0 * tf.log(255.0) / tf.log(10.0) - 10.0 * tf.log(tf.maximum(x, 0.0001)) / tf.log(10.0)
    return x

def trivial(dropped_size):
    """
    Args:
        - dropped_size: number of bytes to drop
    """
    encoder_training = tf.placeholder_with_default(False, [], name='encoder_training')
    decoder_training = tf.placeholder_with_default(False, [], name='decoder_training')
    
    encoder_input = tf.placeholder(tf.uint8, [None, 32, 32, 3], name='encoder_input')
    x = tf.reshape(encoder_input, [-1, 32*32*3])
    x = tf.slice(x, [0,0], [-1,32*32*3-dropped_size])
    encoder_output = tf.identity(x, name='encoder_output')
    
    decoder_input = tf.identity(encoder_output, name='decoder_input')
    x = tf.concat([decoder_input, tf.slice(decoder_input, [0,0], [-1,dropped_size])], 1)
    x = tf.reshape(x, [-1, 32, 32, 3])
    x = tf.cast(x, tf.float32) * tf.Variable(1.0)
    decoder_output = tf.identity(x, name='decoder_output')
    
    return (encoder_input, encoder_output, encoder_training, decoder_input, decoder_output, decoder_training)

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
        numFilters = tf.floor(inputs.get_shape().as_list()[3]*compression)
        x = tf.layers.conv2d(x, numFilters, 1, 1, padding='same')
        x = tf.layers.average_pooling2d(x, 2, 2, padding='same')
    return x
