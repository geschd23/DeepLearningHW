#---------------------------------------------------
# Daniel Geschwender, Yanfeng Liu, Jeevan Rajagopal
# CSCE 496/896: Deep Learning
# Homework 2
#---------------------------------------------------
# Based off code provided in Hackathon 3

import tensorflow as tf
import numpy as np
import math
import os
import util
import model

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/work/cse496dl/shared/homework/02/', 'directory where EMODB/SAVEE is located')
flags.DEFINE_string('save_dir', 'model', 'directory where model graph and weights are saved')
flags.DEFINE_integer('batch_size', 32, '')
flags.DEFINE_integer('max_epoch_num', 200, '')
flags.DEFINE_integer('patience', 10, '')
flags.DEFINE_string('filters', "2", '')
flags.DEFINE_float('learning_rate', 0.0001, '')
flags.DEFINE_float('dropout_rate', 0.0, '')
flags.DEFINE_bool('l2_regularizer', False, '')
flags.DEFINE_bool('output_model', False, '')
flags.DEFINE_float('data_fraction', 1.0, '')
flags.DEFINE_integer('fold', 4, '')
FLAGS = flags.FLAGS

def main(argv):
    print(tf.__version__)
    
    # handle command line arguments
    print("regularizer:",FLAGS.l2_regularizer)
    print("dropout_rate:",FLAGS.dropout_rate)
    print("learning_rate:",FLAGS.learning_rate)
    print("filters:",FLAGS.filters)
    learning_rate = FLAGS.learning_rate
    dropout_rate = FLAGS.dropout_rate
    filters = list(map(int, FLAGS.filters.split(",")))
    regularizer = tf.contrib.layers.l2_regularizer(scale=1.) if FLAGS.l2_regularizer else None
    
    # specify the network
    input = tf.placeholder(tf.float32, [None, 16641], name='input_placeholder')
    input2D = tf.reshape(input, [-1, 129, 129, 1])
    trainingMode = tf.placeholder(tf.bool)
    network = model.conv_block(input2D, filters=filters, dropout_rate=dropout_rate, is_training=trainingMode)
    flat = tf.reshape(network, [-1, 129*129*filters[-1]])
    denseOut = tf.layers.dense(flat, 7)
    output = tf.identity(denseOut, name='output')

    # define classification loss
    label = tf.placeholder(tf.uint8, [None, 7], name='label')
    cross_entropy  = tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=output)
    reduce_mean_cross_entropy = tf.reduce_mean(cross_entropy)
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    REG_COEFF = 0.1
    total_loss = cross_entropy + REG_COEFF * sum(regularization_losses)
    
    # setup confusion matrix and accuracy
    confusion_matrix_op = tf.confusion_matrix(tf.argmax(label, axis=1), tf.argmax(output, axis=1), num_classes=7)
    accuracy = tf.equal(tf.argmax(label, axis=1), tf.argmax(output, axis=1))
    reduce_mean_accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))

    # set up training and saving functionality
    global_step_tensor = tf.get_variable('global_step', trainable=False, shape=[], initializer=tf.zeros_initializer)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    train_op = optimizer.minimize(total_loss, global_step=global_step_tensor)
    saver = tf.train.Saver()
    
    k_fold_accuracy = []
    
    for fold in range(FLAGS.fold):
        print("Beginning fold ", fold)
        
        # load data
        train_images = np.load(FLAGS.data_dir + 'EMODB-German/train_x_' + str(fold+1) + '.npy')
        train_labels = np.load(FLAGS.data_dir + 'EMODB-German/train_y_' + str(fold+1) + '.npy')
        validation_images = np.load(FLAGS.data_dir + 'EMODB-German/test_x_' + str(fold+1) + '.npy')
        validation_labels = np.load(FLAGS.data_dir + 'EMODB-German/test_y_' + str(fold+1) + '.npy')
        train_num_examples = train_images.shape[0]
        validation_num_examples = validation_images.shape[0]
        
        # reduce dataset according to data_fraction parameter
        train_images, _ = util.split_data(train_images, FLAGS.data_fraction)
        train_labels, _ = util.split_data(train_labels, FLAGS.data_fraction)
        validation_images, _ = util.split_data(validation_images, FLAGS.data_fraction)
        validation_labels, _ = util.split_data(validation_labels, FLAGS.data_fraction)

        # set up early stopping
        best_epoch = 0
        best_validation_ce = math.inf
        best_validation_acc = 0
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
                acc_vals = []
                for i in range(train_num_examples // batch_size):
                    batch_xs = train_images[i*batch_size:(i+1)*batch_size, :]
                    batch_ys = train_labels[i*batch_size:(i+1)*batch_size]       
                    _, train_ce, train_acc = session.run([train_op, reduce_mean_cross_entropy, reduce_mean_accuracy], {input: batch_xs, label: batch_ys, trainingMode: True})
                    ce_vals.append(train_ce)
                    acc_vals.append(train_acc)
                avg_train_ce = sum(ce_vals) / len(ce_vals)
                avg_train_acc = sum(acc_vals) / len(acc_vals)
                print('TRAIN CROSS ENTROPY: ' + str(avg_train_ce))
                print('TRAIN ACCURACY: ' + str(train_acc))

                # report mean validation loss
                ce_vals = []
                acc_vals = []
                conf_mxs = []
                for i in range(validation_num_examples // batch_size):
                    batch_xs = validation_images[i*batch_size:(i+1)*batch_size, :]
                    batch_ys = validation_labels[i*batch_size:(i+1)*batch_size]
                    validation_ce, validation_acc, conf_matrix = session.run([reduce_mean_cross_entropy, reduce_mean_accuracy, confusion_matrix_op], {input: batch_xs, label: batch_ys, trainingMode: False})
                    ce_vals.append(validation_ce)
                    acc_vals.append(validation_acc)
                    conf_mxs.append(conf_matrix)
                avg_validation_ce = sum(ce_vals) / len(ce_vals)
                avg_validation_acc = sum(acc_vals) / len(acc_vals)
                print('VALIDATION CROSS ENTROPY: ' + str(avg_validation_ce))
                print('VALIDATION ACCURACY: ' + str(avg_validation_acc))
                print('VALIDATION CONFUSION MATRIX:')
                print(str(sum(conf_mxs)))

                # update best results
                if avg_validation_acc > best_validation_acc:
                    best_epoch = epoch
                    best_validation_ce = avg_validation_ce
                    best_validation_acc = avg_validation_acc
                    wait = 0
                    if FLAGS.output_model:
                        path_prefix = saver.save(session, os.path.join(FLAGS.save_dir, "homework_1"), global_step=0)
                else:
                    wait += 1
                    if wait == patience:
                        break

                # output tensorboard data
                if FLAGS.output_model and epoch==0:
                    file_writer = tf.summary.FileWriter(FLAGS.save_dir, session.graph)

            
            print("Results for fold", fold)
            print('Best Epoch: ' + str(best_epoch))
            print('Best VALIDATION CROSS ENTROPY: ' + str(best_validation_ce))
            print('Best VALIDATION ACCURACY: ' + str(best_validation_acc))
            k_fold_accuracy.append(best_validation_acc)
    
    # report average best accuracy across all k folds
    print('Average accuracy across k folds: '+str(sum(k_fold_accuracy) / len(k_fold_accuracy)))
                
        
if __name__ == "__main__":
    tf.app.run()
