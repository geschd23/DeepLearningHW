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

def dense_conv_block(inputs, filters, dropout_rate, is_training):
    """
    Args:
        - inputs: 4D tensor of shape NHWC
        - filters: iterable of ints
        - dropout_rate: chance of dropping a node
        - is_training: boolean scalar tensor
    """
    with tf.name_scope('conv_block') as scope:
        concatenation = inputs
        for i in range(len(filters)):
            x = concatenation
            x = tf.layers.dropout(x, dropout_rate, training=is_training)
            x = tf.layers.conv2d(x, filters[i], 3, 1, padding='same')
            x = tf.layers.batch_normalization(x, training=is_training)
            x = tf.nn.elu(x)
            concatenation = tf.concat([concatenation, x], 3)
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
            x = tf.layers.dense(linear_nodes[i])
            x = tf.layers.batch_normalization(x, training=is_training)
            x = tf.nn.elu(x)
        x = tf.layers.dense(7)
    return x