import tensorflow as tf

# Based on function provided in Hackathon 5
def conv_block(inputs, filters, dropout_rate, is_training):
    """
    Args:
        - inputs: 4D tensor of shape NHWC
        - filters: iterable of ints
        - dropout_rate: chance of dropping a node
        - is_training: boolean scalar tensor
    """
    with tf.name_scope('conv_block') as scope:
        x = inputs
        for i in range(len(filters)):
            x = tf.layers.dropout(x, dropout_rate, training=is_training)
            x = tf.layers.conv2d(x, filters[i], 3, 1, padding='same')
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
    with tf.name_scope('dense_block') as scope:
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
    with tf.name_scope('transition_layer') as scope:
        x = tf.layers.batch_normalization(inputs, training=is_training)
        numFilters = tf.floor(tf.shape(inputs)[3]*compression)
        x = tf.layers.conv2d(x, numFilters, 1, 1, padding='same')
        x = tf.layers.average_pooling2d(x, 2, 2)
    return x

def classification_end(x, linear_nodes, dropout_rate, is_training):
    """
    Args:
        - x: 4D tensor of shape NHWC; output of another module (conv/dense)
        - linear_nodes: iterable of ints; numbers of nodes per linear layer
        - dropout_rate: chance of dropping a node
        - is_training: boolean scalar tensor
    """
    x = tf.contrib.layers.flatten(x)
    with tf.name_scope('linear') as scope:
        for i in range(len(linear_nodes)):
            x = tf.layers.dropout(x, dropout_rate, training=is_training)
            x = tf.layers.dense(x, linear_nodes[i])
            x = tf.layers.batch_normalization(x, training=is_training)
            x = tf.nn.elu(x)
        x = tf.layers.dense(x, 7)
    return x