import tensorflow as tf
import numpy as np

def dense(input, nodes, regularizer, dropout_rate, training) : 
    x = tf.layers.dropout(input, dropout_rate, training=training)
    x = tf.layers.dense(x, nodes)
    x = tf.nn.elu(x)
    return x

def convolution(input, filters, size, regularizer, dropout_rate, training) : 
    x = tf.layers.dropout(input, dropout_rate, training=training)
    x = tf.layers.conv2d(x, filters, size, 1, padding='same', kernel_regularizer=regularizer, bias_regularizer=regularizer)
    x = tf.nn.elu(x)
    return x

def sc2network():
    screen_input = tf.placeholder(tf.float32, [None, 64, 64, 17], name='screen_input')
    minimap_input = tf.placeholder(tf.float32, [None, 64, 64, 7], name='minimap_input')
    player_input = tf.placeholder(tf.float32, [None, 11], name='player_input')
    single_select_input = tf.placeholder(tf.float32, [None, 7], name='single_select_input')
    action_mask = tf.placeholder(tf.float32, [None, 524], name='action_mask')
    
    regularizer = tf.contrib.layers.l2_regularizer(scale=1.)
    dropout_rate = 0.0
    training = tf.placeholder_with_default(False, [], name='training_flag')
    
    with tf.variable_scope('sc2network'):
        with tf.variable_scope('screen'):
            screen = screen_input
            screen = convolution(screen, 16, 5, regularizer, dropout_rate, training)
            screen = convolution(screen, 32, 3, regularizer, dropout_rate, training)
            
        with tf.variable_scope('minimap'):
            minimap = minimap_input
            minimap = convolution(minimap, 16, 5, regularizer, dropout_rate, training)
            minimap = convolution(minimap, 32, 3, regularizer, dropout_rate, training)
            
        with tf.variable_scope('state'):
            broadcast_player_input = tf.tile(tf.reshape(player_input, [-1,1,1,11]), [1, 64, 64, 1])
            broadcast_single_select_input = tf.tile(tf.reshape(single_select_input, [-1,1,1,7]), [1, 64, 64, 1])
            full_state = tf.concat([screen, minimap, broadcast_player_input, broadcast_single_select_input], axis=3)
            dense_state = dense(tf.contrib.layers.flatten(full_state), 256, regularizer, dropout_rate, training)
            
        with tf.variable_scope('value'):
            value = dense(dense_state, 1, regularizer, dropout_rate, training)
            
        with tf.variable_scope('action'):
            action_logits = dense(dense_state, 524, regularizer, dropout_rate, training)
            masked_action_logits = action_logits-1e32*(1-action_mask)
            action = tf.multinomial(masked_action_logits, 1)
            
        with tf.variable_scope('param_screen'):
            param_screen_logits = convolution(full_state, 1, 1, regularizer, dropout_rate, training)
            param_screen = tf.multinomial(tf.contrib.layers.flatten(param_screen_logits), 1)
            param_screen = tf.concat([param_screen//64, param_screen%64], axis=1)
            
        with tf.variable_scope('param_minimap'):
            param_minimap_logits = convolution(full_state, 1, 1, regularizer, dropout_rate, training)
            param_minimap = tf.multinomial(tf.contrib.layers.flatten(param_minimap_logits), 1)
            param_minimap = tf.concat([param_minimap//64, param_minimap%64], axis=1)
            
        with tf.variable_scope('param_screen2'):
            param_screen2_logits = convolution(full_state, 1, 1, regularizer, dropout_rate, training)
            param_screen2 = tf.multinomial(tf.contrib.layers.flatten(param_screen2_logits), 1)
            param_screen2 = tf.concat([param_screen2//64, param_screen2%64], axis=1)
            
        with tf.variable_scope('param_queued'):
            param_queued_logits = dense(dense_state, 2, regularizer, dropout_rate, training)
            param_queued = tf.multinomial(param_queued_logits, 1)
            
        with tf.variable_scope('param_control_group_act'):
            param_control_group_act_logits = dense(dense_state, 5, regularizer, dropout_rate, training)
            param_control_group_act = tf.multinomial(param_control_group_act_logits, 1)
            
        with tf.variable_scope('param_control_group_id'):
            param_control_group_id_logits = dense(dense_state, 10, regularizer, dropout_rate, training)
            param_control_group_id = tf.multinomial(param_control_group_id_logits, 1)
            
        with tf.variable_scope('param_select_point_act'):
            param_select_point_act_logits = dense(dense_state, 4, regularizer, dropout_rate, training)
            param_select_point_act = tf.multinomial(param_select_point_act_logits, 1)
            
        with tf.variable_scope('param_select_add'):
            param_select_add_logits = dense(dense_state, 2, regularizer, dropout_rate, training)
            param_select_add = tf.multinomial(param_select_add_logits, 1)
            
        with tf.variable_scope('param_select_unit_act'):
            param_select_unit_act_logits = dense(dense_state, 4, regularizer, dropout_rate, training)
            param_select_unit_act = tf.multinomial(param_select_unit_act_logits, 1)
            
        with tf.variable_scope('param_select_unit_id'):
            param_select_unit_id_logits = dense(dense_state, 500, regularizer, dropout_rate, training)
            param_select_unit_id = tf.multinomial(param_select_unit_id_logits, 1)
            
        with tf.variable_scope('param_select_worker'):
            param_select_worker_logits = dense(dense_state, 4, regularizer, dropout_rate, training)
            param_select_worker = tf.multinomial(param_select_worker_logits, 1)
            
        with tf.variable_scope('param_build_queue_id'):
            param_build_queue_id_logits = dense(dense_state, 10, regularizer, dropout_rate, training)
            param_build_queue_id = tf.multinomial(param_build_queue_id_logits, 1)
            
        with tf.variable_scope('param_unload_id'):
            param_unload_id_logits = dense(dense_state, 500, regularizer, dropout_rate, training)
            param_unload_id = tf.multinomial(param_unload_id_logits, 1)
            
        param_list = [param_screen, param_minimap, param_screen2, param_queued,
                      param_control_group_act, param_control_group_id,
                      param_select_point_act, param_select_add, param_select_unit_act,
                      param_select_unit_id, param_select_worker, param_build_queue_id,
                      param_unload_id]
        
    return (screen_input, minimap_input, player_input, single_select_input, action_mask, action, param_list)
