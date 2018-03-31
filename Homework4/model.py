import tensorflow as tf
import numpy as np

def sentence_completion_rnn(encoder_input, decoder_input, layer_nodes, batch_size):
    
    # encoder
    with tf.variable_scope('encoder'):
        encoder_layers = [tf.nn.rnn_cell.LSTMCell(nodes) for nodes in layer_nodes]
        encoder_multicell = tf.nn.rnn_cell.MultiRNNCell(encoder_layers)
        encoder_output, encoder_state = tf.nn.dynamic_rnn(cell=encoder_multicell,
                                           inputs=encoder_input,
                                           dtype=tf.float32)
    
    # decoder
    with tf.variable_scope('decoder'):
        decoder_layers = [tf.nn.rnn_cell.LSTMCell(nodes) for nodes in layer_nodes]
        decoder_multicell = tf.nn.rnn_cell.MultiRNNCell(decoder_layers)
        decoder_output, decoder_state = tf.nn.dynamic_rnn(cell=decoder_multicell,
                                           inputs=decoder_input,
                                           initial_state=encoder_state,
                                           dtype=tf.float32)
    
    x = tf.layers.dense(decoder_output, encoder_input.get_shape().as_list()[2])
    x = tf.tanh(x)
    x = x / tf.norm(x, axis=2, keep_dims=True)
    return x