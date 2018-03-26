import tensorflow as tf
import numpy as np

def lstm(input, layer_nodes, batch_size):
    layers = [tf.nn.rnn_cell.LSTMCell(nodes) for nodes in layer_nodes]
    multicell = tf.nn.rnn_cell.MultiRNNCell(layers)
    x, state = tf.nn.dynamic_rnn(cell=multicell,
                                       inputs=input,
                                       dtype=tf.float32)
    x = tf.layers.dense(x, input.get_shape().as_list()[2])
    x = tf.tanh(x)
    x = x / tf.norm(x, axis=2, keep_dims=True)
    return x