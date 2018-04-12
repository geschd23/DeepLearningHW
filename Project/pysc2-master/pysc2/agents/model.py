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

def sc2network(optimizer, beta, eta, scope):
    with tf.variable_scope(scope):
        regularizer = tf.contrib.layers.l2_regularizer(scale=1.)
        dropout_rate = 0.0
        training = tf.placeholder_with_default(False, [], name='training_flag')
        
        
        with tf.variable_scope('observation'):
            
            screen_input = tf.placeholder(tf.float32, [None, 64, 64, 17], name='screen_input')
            minimap_input = tf.placeholder(tf.float32, [None, 64, 64, 7], name='minimap_input')
            player_input = tf.placeholder(tf.float32, [None, 11], name='player_input')
            single_select_input = tf.placeholder(tf.float32, [None, 7], name='single_select_input')
            action_mask = tf.placeholder(tf.float32, [None, 524], name='action_mask')

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
            
        
        with tf.variable_scope('policy'):

            with tf.variable_scope('action'):
                action_logits = dense(dense_state, 524, regularizer, dropout_rate, training)
                masked_action_logits = action_logits-1e32*(1-action_mask)
                action_policy = tf.nn.softmax(masked_action_logits)

            with tf.variable_scope('param_screen'):
                param_screen_logits = convolution(full_state, 1, 1, regularizer, dropout_rate, training)
                param_screen_logits = tf.contrib.layers.flatten(param_screen_logits)
                param_screen_policy = tf.nn.softmax(param_screen_logits)

            with tf.variable_scope('param_minimap'):
                param_minimap_logits = convolution(full_state, 1, 1, regularizer, dropout_rate, training)
                param_minimap_logits = tf.contrib.layers.flatten(param_minimap_logits)
                param_minimap_policy = tf.nn.softmax(param_minimap_logits)

            with tf.variable_scope('param_screen2'):
                param_screen2_logits = convolution(full_state, 1, 1, regularizer, dropout_rate, training)
                param_screen2_logits = tf.contrib.layers.flatten(param_screen2_logits)
                param_screen2_policy = tf.nn.softmax(param_screen2_logits)

            with tf.variable_scope('param_queued'):
                param_queued_logits = dense(dense_state, 2, regularizer, dropout_rate, training)
                param_queued_policy = tf.nn.softmax(param_queued_logits)

            with tf.variable_scope('param_control_group_act'):
                param_control_group_act_logits = dense(dense_state, 5, regularizer, dropout_rate, training)
                param_control_group_act_policy = tf.nn.softmax(param_control_group_act_logits)

            with tf.variable_scope('param_control_group_id'):
                param_control_group_id_logits = dense(dense_state, 10, regularizer, dropout_rate, training)
                param_control_group_id_policy = tf.nn.softmax(param_control_group_id_logits)

            with tf.variable_scope('param_select_point_act'):
                param_select_point_act_logits = dense(dense_state, 4, regularizer, dropout_rate, training)
                param_select_point_act_policy = tf.nn.softmax(param_select_point_act_logits)

            with tf.variable_scope('param_select_add'):
                param_select_add_logits = dense(dense_state, 2, regularizer, dropout_rate, training)
                param_select_add_policy = tf.nn.softmax(param_select_add_logits)

            with tf.variable_scope('param_select_unit_act'):
                param_select_unit_act_logits = dense(dense_state, 4, regularizer, dropout_rate, training)
                param_select_unit_act_policy = tf.nn.softmax(param_select_unit_act_logits)

            with tf.variable_scope('param_select_unit_id'):
                param_select_unit_id_logits = dense(dense_state, 500, regularizer, dropout_rate, training)
                param_select_unit_id_policy = tf.nn.softmax(param_select_unit_id_logits)

            with tf.variable_scope('param_select_worker'):
                param_select_worker_logits = dense(dense_state, 4, regularizer, dropout_rate, training)
                param_select_worker_policy = tf.nn.softmax(param_select_worker_logits)

            with tf.variable_scope('param_build_queue_id'):
                param_build_queue_id_logits = dense(dense_state, 10, regularizer, dropout_rate, training)
                param_build_queue_id_policy = tf.nn.softmax(param_build_queue_id_logits)

            with tf.variable_scope('param_unload_id'):
                param_unload_id_logits = dense(dense_state, 500, regularizer, dropout_rate, training)
                param_unload_id_policy = tf.nn.softmax(param_unload_id_logits)

            param_policy = [param_screen_policy, param_minimap_policy, param_screen2_policy, param_queued_policy,
                          param_control_group_act_policy, param_control_group_id_policy,
                          param_select_point_act_policy, param_select_add_policy, param_select_unit_act_policy,
                          param_select_unit_id_policy, param_select_worker_policy, param_build_queue_id_policy,
                          param_unload_id_policy]

            
        if scope != 'global':
            with tf.variable_scope('loss'):

                with tf.variable_scope('loss_inputs'):
                    advantage_input = tf.placeholder(tf.float32, [None, 1], name='advantage_input')
                    target_value_input = tf.placeholder(tf.float32, [None, 1], name='target_value_input')

                with tf.variable_scope('action_inputs'):
                    action_input = tf.placeholder(tf.int32, [None, 1], name='action_input')
                    param_screen_input = tf.placeholder(tf.int32, [None, 1], name='param_screen_input')
                    param_minimap_input = tf.placeholder(tf.int32, [None, 1], name='param_minimap_input')
                    param_screen2_input = tf.placeholder(tf.int32, [None, 1], name='param_screen2_input')
                    param_queued_input = tf.placeholder(tf.int32, [None, 1], name='param_queued_input')
                    param_control_group_act_input = tf.placeholder(tf.int32, [None, 1], name='param_control_group_act_input')
                    param_control_group_id_input = tf.placeholder(tf.int32, [None, 1], name='param_control_group_id_input')
                    param_select_point_act_input = tf.placeholder(tf.int32, [None, 1], name='param_select_point_act_input')
                    param_select_add_input = tf.placeholder(tf.int32, [None, 1], name='param_select_add_input')
                    param_select_unit_act_input = tf.placeholder(tf.int32, [None, 1], name='param_select_unit_act_input')
                    param_select_unit_id_input = tf.placeholder(tf.int32, [None, 1], name='param_select_unit_id_input')
                    param_select_worker_input = tf.placeholder(tf.int32, [None, 1], name='param_select_worker_input')
                    param_build_queue_id_input = tf.placeholder(tf.int32, [None, 1], name='param_build_queue_id_input')
                    param_unload_id_input = tf.placeholder(tf.int32, [None, 1], name='param_unload_id_input')

                    param_input = [param_screen_input, param_minimap_input, param_screen2_input, param_queued_input,
                              param_control_group_act_input, param_control_group_id_input,
                              param_select_point_act_input, param_select_add_input, param_select_unit_act_input,
                              param_select_unit_id_input, param_select_worker_input, param_build_queue_id_input,
                              param_unload_id_input]

                with tf.variable_scope('policy_loss'):
                    action_policy_used = tf.reduce_sum(tf.one_hot(action_input, 524) * action_policy, axis=1)
                    param_policy_used = [ tf.reduce_sum(tf.one_hot(param_input[i], param_policy[i].get_shape().as_list()[-1]) * param_policy[i], axis=1) for i in range(len(param_policy))]
                    policy_loss = tf.reduce_sum(advantage_input * tf.log(tf.concat([action_policy_used]+param_policy_used, axis=1)+1E-5))

                with tf.variable_scope('entropy_loss'):
                    action_entropy = action_policy * tf.log(action_policy+1E-5)
                    param_entropy = [ param_policy[i] * tf.log(param_policy[i]+1E-5) for i in range(len(param_policy))]
                    entropy_loss = tf.reduce_sum(tf.concat([action_entropy]+param_entropy, axis=1))

                with tf.variable_scope('value_loss'):
                    value_loss = tf.reduce_sum(tf.square(target_value_input-value))

                with tf.variable_scope('total_loss'):
                    total_loss = -policy_loss + beta*value_loss + eta*entropy_loss

                with tf.variable_scope('gradients'):
                    local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                    gradients = tf.gradients(total_loss, local_vars)
                    global_norm = tf.global_norm(local_vars)
                    clipped_gradients, clipped = tf.clip_by_global_norm(gradients,1.0)
                    update_step = optimizer.apply_gradients(zip(clipped_gradients, local_vars))
                        
        
    return (screen_input, minimap_input, player_input, single_select_input, action_mask, action_policy, param_policy, value, action_input, param_input, advantage_input, target_value_input, gradients, update_step, global_norm, clipped)
