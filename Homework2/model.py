import tensorflow as tf

def create_model(input, trainingMode, architecture, regularizer, dropout_rate):
    """
    Generates a block of tensors

    Args:
        - input: the input tensor
        - trainingMode: a scalar tensor to toggle training mode for dropout
        - architecture: array indicating the number of nodes to include in dense layers
        - regularizer: the regularization function to use
        - dropout_rate: probability of dropping a node during dropout
    """
    # normalize the input
    layers = [tf.divide(input,255.0)]
    
    with tf.name_scope('linear_model') as scope:
        # dropout on input
        layers.append(tf.layers.Dropout(dropout_rate))
        
        # construct each hidden layer
        for nodes in architecture:
            layers.append(tf.layers.Dense(nodes,
                             kernel_regularizer=regularizer,
                             bias_regularizer=regularizer,
                             activation=tf.nn.relu))
            layers.append(tf.layers.Dropout(dropout_rate))
            
        # construct the output layer
        layers.append(tf.layers.Dense(10,
                             kernel_regularizer=regularizer,
                             bias_regularizer=regularizer))

        # construct tensors from tensor objects
        for i in range(len(layers)-1):
            if i%2==0:  # dropout layer
                layers[i+1]=layers[i+1](layers[i], training=trainingMode)
            else: # non-dropout layer
                layers[i+1]=layers[i+1](layers[i])
                
    return layers[len(layers)-1]
