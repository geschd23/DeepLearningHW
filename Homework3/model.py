import tensorflow as tf
import numpy as np

def encoder(code_size):
    input = tf.placeholder(tf.float32, [None, 32, 32, 3], name='encoder_input')
    training = tf.placeholder_with_default(False, [], name='encoder_training')
    
    with tf.variable_scope('encoder'):
        x = input
        x = tf.layers.max_pooling2d(x, 2, 2, padding='same')
        flatten_dim = np.prod(x.get_shape().as_list()[1:])
        x = tf.reshape(x, [-1, flatten_dim])
        x = tf.layers.dense(x, code_size)
        x = tf.nn.elu(x)
        
    output = tf.identity(x, name='encoder_output')
    return (input, output, training, flatten_dim)

def decoder(code_size, encoder_output, flatten_dim):
    input = tf.identity(encoder_output, name='decoder_input')
    training = tf.placeholder_with_default(False, [], name='decoder_training')
    
    with tf.variable_scope('decoder'):
        x = input
        x = tf.layers.dense(x, flatten_dim)
        x = tf.nn.elu(x)
        x = tf.reshape(x, [-1, 16, 16, 3])
        x = tf.layers.conv2d_transpose(x, 3, 3, strides=(2, 2), padding='same')
        
    output = tf.identity(x, name='decoder_output')
    return (input, output, training)

# Based on function provided in Hackathon 5
def conv_block(inputs, filters, regularizer, dropout_rate, is_training):
    """
    Args:
        - inputs: 4D tensor of shape NHWC
        - filters: iterable of ints
        - regularizer: the regularizer to use
        - dropout_rate: chance of dropping a node
        - is_training: boolean scalar tensor
    """
    with tf.variable_scope('conv_block'):
        x = inputs
        for i in range(len(filters)):
            x = tf.layers.dropout(x, dropout_rate, training=is_training)
            if filters[i]==0:
                x = tf.layers.max_pooling2d(x, 2, 2, padding='same')
            else:
                x = tf.layers.conv2d(x, filters[i], 3, 1, padding='same', kernel_regularizer=regularizer, bias_regularizer=regularizer)
                x = tf.layers.batch_normalization(x, training=is_training)
                x = tf.nn.elu(x)
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

def psnr(original, reconstruction):
    """
    Args:
        - original: 4D tensor of shape NHWC, the original image
        - reconstruction: 4D tensor of shape NHWC, the reconstructed image
    """
    with tf.variable_scope('psnr') as scope:
        #x = tf.losses.mean_squared_error(original, reconstruction)
        x = tf.losses.mean_squared_error(original, reconstruction, reduction = tf.losses.Reduction.NONE)
        x = tf.reduce_mean(x, axis=3)
        x = tf.reduce_mean(x, axis=2)
        x = tf.reduce_mean(x, axis=1)
        x = 20.0 * tf.log(255.0) / tf.log(10.0) - 10.0 * tf.log(x) / tf.log(10.0)
    return x
