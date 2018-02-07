import tensorflow as tf

def create_model(input):
    """
    Generates a block of tensors
    
    Args:
        - input: the input tensor
    """
    with tf.name_scope('linear_model') as scope:
        hidden1 = tf.layers.Dense(256,
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.),
                             bias_regularizer=tf.contrib.layers.l2_regularizer(scale=1.),
                             activation=tf.nn.relu)
        hidden2 = tf.layers.Dense(128,
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.),
                             bias_regularizer=tf.contrib.layers.l2_regularizer(scale=1.),
                             activation=tf.nn.relu)
        hidden3 = tf.layers.Dense(256,
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.),
                             bias_regularizer=tf.contrib.layers.l2_regularizer(scale=1.),
                             activation=tf.nn.relu)                     
        output = tf.layers.Dense(10,
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.),
                             bias_regularizer=tf.contrib.layers.l2_regularizer(scale=1.))
        return output(hidden3(hidden2(hidden1(input))))