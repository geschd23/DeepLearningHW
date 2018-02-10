import tensorflow as tf

def create_model(input, architecture, regularizer, keep_probability):
    """
    Generates a block of tensors

    Args:
        - input: the input tensor
        - architecture: array indicating the number of nodes to include in dense layers
        - regularizer: the regularization function to use
        - keep_probablity: probability of keeping a node during dropout
    """
    # normalize the input
    layers = [tf.divide(input,255.0)]
    
    with tf.name_scope('linear_model') as scope:
        # dropout on input
        layers.append(tf.layers.Dropout(keep_probability))
        
        # construct each hidden layer
        for nodes in architecture:
            layers.append(tf.layers.Dense(nodes,
                             kernel_regularizer=regularizer,
                             bias_regularizer=regularizer,
                             activation=tf.nn.relu))
            layers.append(tf.layers.Dropout(keep_probability))
            
        # construct the output layer
        layers.append(tf.layers.Dense(10,
                             kernel_regularizer=regularizer,
                             bias_regularizer=regularizer))

        # construct tensors from tensor objects
        for i in range(len(layers)-1):
            layers[i+1]=layers[i+1](layers[i])

    return layers[len(layers)-1]
