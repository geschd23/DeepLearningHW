#------------------------------------------
# Daniel Geschwender and Yanfeng Liu
# CSCE 496/896: Deep Learning
# Homework 1
#------------------------------------------
# Based off code provided in Hackathon 3

import tensorflow as tf
import numpy as np
import math
import os
import util
import model

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/work/cse496dl/shared/homework/01/', 'directory where FMNIST is located')
flags.DEFINE_string('save_dir', 'model', 'directory where model graph and weights are saved')
flags.DEFINE_integer('batch_size', 32, '')
flags.DEFINE_integer('max_epoch_num', 100, '')
flags.DEFINE_integer('patience', 5, '')
flags.DEFINE_float('data_fraction', 1.0, '')
FLAGS = flags.FLAGS

def main(argv):
    print(tf.__version__)
    
    # load data
    train_images = np.load(FLAGS.data_dir + 'fmnist_train_data.npy')
    train_labels = np.load(FLAGS.data_dir + 'fmnist_train_labels.npy')

    # reduce dataset according to data_fraction parameter
    train_images, _ = util.split_data(train_images, FLAGS.data_fraction)
    train_labels, _ = util.split_data(train_labels, FLAGS.data_fraction)
    
    # split into train and validate
    train_images, validation_images = util.split_data(train_images, 0.9)
    train_labels, validation_labels = util.split_data(train_labels, 0.9)
    train_num_examples = train_images.shape[0]
    validation_num_examples = validation_images.shape[0]

    # specify the network
    input = tf.placeholder(tf.float32, [None, 784], name='input_placeholder')
    network = model.create_model(input)
    output = tf.identity(network, name='output')

    # define classification loss
    label = tf.placeholder(tf.uint8, [None], name='label')
    oneHot = tf.one_hot(label,10)
    cross_entropy  = tf.nn.softmax_cross_entropy_with_logits(labels=oneHot, logits=output)
    reduce_mean_cross_entropy = tf.reduce_mean(cross_entropy)
    
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    # this is the weight of the regularization part of the final loss
    REG_COEFF = 0.1
    # this value is what we'll pass to `minimize`
    total_loss = cross_entropy + REG_COEFF * sum(regularization_losses)
    
    confusion_matrix_op = tf.confusion_matrix(tf.argmax(oneHot, axis=1), tf.argmax(output, axis=1), num_classes=10)

    # set up training and saving functionality
    global_step_tensor = tf.get_variable('global_step', trainable=False, shape=[], initializer=tf.zeros_initializer)
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001)
    train_op = optimizer.minimize(total_loss, global_step=global_step_tensor)
    saver = tf.train.Saver()
    
    # set up early stopping
    best_epoch = 0
    best_train_ce = math.inf
    best_validation_ce = math.inf
    best_test_ce = math.inf
    patience = FLAGS.patience
    wait = 0
    
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        # run training
        batch_size = FLAGS.batch_size
        for epoch in range(FLAGS.max_epoch_num):
            print('Epoch: ' + str(epoch))

            # run gradient steps and report mean loss on train data
            ce_vals = []
            for i in range(train_num_examples // batch_size):
                batch_xs = train_images[i*batch_size:(i+1)*batch_size, :]
                batch_ys = train_labels[i*batch_size:(i+1)*batch_size]       
                _, train_ce = session.run([train_op, reduce_mean_cross_entropy], {input: batch_xs, label: batch_ys})
                ce_vals.append(train_ce)
            avg_train_ce = sum(ce_vals) / len(ce_vals)
            print('TRAIN CROSS ENTROPY: ' + str(avg_train_ce))

            # report mean validation loss
            ce_vals = []
            conf_mxs = []
            for i in range(validation_num_examples // batch_size):
                batch_xs = validation_images[i*batch_size:(i+1)*batch_size, :]
                batch_ys = validation_labels[i*batch_size:(i+1)*batch_size]
                validation_ce, conf_matrix = session.run([reduce_mean_cross_entropy, confusion_matrix_op], {input: batch_xs, label: batch_ys})
                ce_vals.append(validation_ce)
                conf_mxs.append(conf_matrix)
            avg_validation_ce = sum(ce_vals) / len(ce_vals)
            print('VALIDATION CROSS ENTROPY: ' + str(avg_validation_ce))
            print('VALIDATION CONFUSION MATRIX:')
            print(str(sum(conf_mxs)))
            
            if avg_validation_ce < best_validation_ce:
                best_epoch = epoch
                best_train_ce = avg_train_ce
                best_validation_ce = avg_validation_ce
                wait = 0
                path_prefix = saver.save(session, os.path.join(FLAGS.save_dir, "homework_1"), global_step=0)
            else:
                wait += 1
                if wait == patience:
                    break

        print('Best Epoch: ' + str(best_epoch))
        print('Best TRAIN CROSS ENTROPY: ' + str(best_train_ce))
        print('Best VALIDATION CROSS ENTROPY: ' + str(best_validation_ce))
        #path_prefix = saver.save(session, os.path.join(FLAGS.save_dir, "mnist_inference"), global_step=global_step_tensor)
        
if __name__ == "__main__":
    tf.app.run()
