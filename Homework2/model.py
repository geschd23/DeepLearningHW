import tensorflow as tf
import numpy as np

def original_model(filters, linear_nodes, regularizer, dropout_rate):
    """
    Args:
        - filters: iterable of ints
        - linear_nodes: iterable of ints; numbers of nodes per linear layer
        - regularizer: the regularizer to use
        - dropout_rate: chance of dropping a node
        - is_training: boolean scalar tensor
    """
    input = tf.placeholder(tf.float32, [None, 16641], name='input_placeholder')
    is_training = tf.placeholder(tf.bool, name='is_training')
    input2D = tf.reshape(input, [-1, 129, 129, 1])
    conv_module = conv_block(input2D, filters=filters, regularizer=regularizer, dropout_rate=dropout_rate, is_training=is_training)
    conv_out = tf.identity(conv_module, name='transfer_point')
    denseOut = classification_end(conv_out, linear_nodes=linear_nodes, regularizer=regularizer, dropout_rate=dropout_rate, is_training=is_training)
    output = tf.identity(denseOut, name='output')
    return (input, output, is_training)
    
    
def transfer_model(transfer, filters, linear_nodes, regularizer, dropout_rate):
    """
    Args:
        - transfer: model file to transfer from
        - filters: iterable of ints
        - linear_nodes: iterable of ints; numbers of nodes per linear layer
        - regularizer: the regularizer to use
        - dropout_rate: chance of dropping a node
        - is_training: boolean scalar tensor
    """
    session = tf.Session()
    saver = tf.train.import_meta_graph(transfer + '.meta')
    saver.restore(session, transfer)
    graph = session.graph
    input = graph.get_tensor_by_name('input_placeholder:0')
    is_training = graph.get_tensor_by_name('is_training:0')
    conv_out = graph.get_tensor_by_name('transfer_point:0')
    stop_grad = tf.stop_gradient(conv_out)
    denseOut = classification_end(stop_grad, linear_nodes=linear_nodes, regularizer=regularizer, dropout_rate=dropout_rate, is_training=is_training)
    output = tf.identity(denseOut, name='output2')
    return (input, output, is_training)


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
    with tf.name_scope('conv_block') as scope:
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


def classification_end(x, linear_nodes, regularizer, dropout_rate, is_training):
    """
    Args:
        - x: 4D tensor of shape NHWC; output of another module (conv/dense)
        - linear_nodes: iterable of ints; numbers of nodes per linear layer
        - regularizer: the regularizer to use
        - dropout_rate: chance of dropping a node
        - is_training: boolean scalar tensor
    """
    flatten_dim = np.prod(x.get_shape().as_list()[1:])
    x = tf.reshape(x, [-1, flatten_dim])
    with tf.name_scope('linear') as scope:
        for i in range(len(linear_nodes)):
            x = tf.layers.dropout(x, dropout_rate, training=is_training)
            x = tf.layers.dense(x, linear_nodes[i], kernel_regularizer=regularizer, bias_regularizer=regularizer)
            x = tf.layers.batch_normalization(x, training=is_training)
            x = tf.nn.elu(x)
        x = tf.layers.dense(x, 7)
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
        x = tf.layers.average_pooling2d(x, 2, 2, padding='same')
    return x
