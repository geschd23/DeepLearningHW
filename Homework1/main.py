#---------------------------------------------------
# Daniel Geschwender, Yanfeng Liu, Jeevan Rajagopal
# CSCE 496/896: Deep Learning
# Homework 1
#---------------------------------------------------
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
flags.DEFINE_integer('max_epoch_num', 200, '')
flags.DEFINE_integer('patience', 10, '')
flags.DEFINE_string('architecture', "50", '')
flags.DEFINE_float('learning_rate', 0.0001, '')
flags.DEFINE_float('dropout_rate', 0.0, '')
flags.DEFINE_bool('l2_regularizer', False, '')
flags.DEFINE_bool('output_model', False, '')
flags.DEFINE_float('data_fraction', 1.0, '')
flags.DEFINE_integer('num_folds', 10, '')
FLAGS = flags.FLAGS

def main(argv):
    print(tf.__version__)
    
    # handle command line arguments
    print("regularizer:",FLAGS.l2_regularizer)
    print("dropout_rate:",FLAGS.dropout_rate)
    print("learning_rate:",FLAGS.learning_rate)
    print("architecture:",FLAGS.architecture)
    learning_rate = FLAGS.learning_rate
    dropout_rate = FLAGS.dropout_rate
    architecture = list(map(int, FLAGS.architecture.split(",")))
    regularizer = tf.contrib.layers.l2_regularizer(scale=1.) if FLAGS.l2_regularizer else None
    
    # specify the network
    input = tf.placeholder(tf.float32, [None, 784], name='input_placeholder')
    trainingMode = tf.placeholder(tf.bool)
    network = model.create_model(input, trainingMode, architecture=architecture, regularizer=regularizer, dropout_rate=dropout_rate)
    output = tf.identity(network, name='output')

    # define classification loss
    label = tf.placeholder(tf.uint8, [None], name='label')
    oneHot = tf.one_hot(label,10)
    cross_entropy  = tf.nn.softmax_cross_entropy_with_logits(labels=oneHot, logits=output)
    reduce_mean_cross_entropy = tf.reduce_mean(cross_entropy)
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    REG_COEFF = 0.1
    total_loss = cross_entropy + REG_COEFF * sum(regularization_losses)

    # setup confusion matrix and accuracy
    confusion_matrix_op = tf.confusion_matrix(tf.argmax(oneHot, axis=1), tf.argmax(output, axis=1), num_classes=10)
    accuracy = tf.equal(tf.argmax(oneHot, axis=1), tf.argmax(output, axis=1))
    reduce_mean_accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))

    # set up training and saving functionality
    global_step_tensor = tf.get_variable('global_step', trainable=False, shape=[], initializer=tf.zeros_initializer)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    train_op = optimizer.minimize(total_loss, global_step=global_step_tensor)
    saver = tf.train.Saver()
    
    k_fold_accuracy = []
    
    for fold in range(FLAGS.num_folds):
        print("Beginning fold ", fold)
        
        # load data
        train_images = np.load(FLAGS.data_dir + 'fmnist_train_data.npy')
        train_labels = np.load(FLAGS.data_dir + 'fmnist_train_labels.npy')

        # reduce dataset according to data_fraction parameter
        train_images, _ = util.split_data(train_images, FLAGS.data_fraction)
        train_labels, _ = util.split_data(train_labels, FLAGS.data_fraction)

        # split into train and validate
        train_images, validation_images = util.k_fold_split(train_images, FLAGS.num_folds, fold)
        train_labels, validation_labels = util.k_fold_split(train_labels, FLAGS.num_folds, fold)
        train_num_examples = train_images.shape[0]
        validation_num_examples = validation_images.shape[0]

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
