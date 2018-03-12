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
flags.DEFINE_string('dataset', 'EMODB-German', 'dataset to run on')
flags.DEFINE_string('model_transfer', '', 'Where to load model to transfer from')
flags.DEFINE_integer('batch_size', 32, '')
flags.DEFINE_integer('max_epoch_num', 50, '')
flags.DEFINE_integer('patience', 0, '')
flags.DEFINE_string('filters', "2,0,4,0", '')
flags.DEFINE_string('linear_nodes', "", '')
flags.DEFINE_float('learning_rate', 0.001, '')
flags.DEFINE_float('dropout_rate', 0.2, '')
flags.DEFINE_float('l2_regularizer', 0.0, '')
flags.DEFINE_bool('output_model', False, '')
flags.DEFINE_float('data_fraction', 1.0, '')
flags.DEFINE_integer('fold', 0, '')
FLAGS = flags.FLAGS

def main(argv):
    print(tf.__version__)

    # handle command line arguments
    print("regularizer:",FLAGS.l2_regularizer)
    print("dropout_rate:",FLAGS.dropout_rate)
    print("learning_rate:",FLAGS.learning_rate)
    print("filters:",FLAGS.filters)
    print("linear_nodes:",FLAGS.linear_nodes)
    learning_rate = FLAGS.learning_rate
    dropout_rate = FLAGS.dropout_rate
    filters = list(map(int, FLAGS.filters.split(","))) if FLAGS.filters != "" else []
    linear_nodes = list(map(int, FLAGS.linear_nodes.split(","))) if FLAGS.linear_nodes != "" else []
    regularizer = tf.contrib.layers.l2_regularizer(scale=1.)
    folds = range(1,5) if FLAGS.fold == 0 else [FLAGS.fold]
    modelFile = "emodb_homework_2" if FLAGS.dataset == "EMODB-German" else "savee_homework_2"


    # specify the network
    if FLAGS.model_transfer == "":
        input, output, trainingMode = model.original_model(filters=filters, linear_nodes=linear_nodes, regularizer=regularizer, dropout_rate=dropout_rate)
    else:
        input, output, trainingMode = model.transfer_model(transfer=FLAGS.model_transfer, filters=filters, linear_nodes=linear_nodes, regularizer=regularizer, dropout_rate=dropout_rate)

    # define classification loss
    label = tf.placeholder(tf.uint8, [None, 7], name='label')
    cross_entropy  = tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=output)
    reduce_mean_cross_entropy = tf.reduce_mean(cross_entropy)
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    REG_COEFF = FLAGS.l2_regularizer
    total_loss = reduce_mean_cross_entropy + REG_COEFF * sum(regularization_losses)

    # setup confusion matrix and accuracy
    confusion_matrix_op = tf.confusion_matrix(tf.argmax(label, axis=1), tf.argmax(output, axis=1), num_classes=7)
    accuracy = tf.equal(tf.argmax(label, axis=1), tf.argmax(output, axis=1))
    reduce_mean_accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))

    # set up training and saving functionality
    global_step_tensor = tf.get_variable('global_step', trainable=False, shape=[], initializer=tf.zeros_initializer)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    var_list = None if FLAGS.model_transfer == "" else tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "new_layers")
    train_op = optimizer.minimize(total_loss, global_step=global_step_tensor, var_list=var_list)
    saver = tf.train.Saver()

    k_fold_accuracy = []

    for fold in folds:
        print("Beginning fold ", fold)

        # load data
        train_images = np.load(FLAGS.data_dir + FLAGS.dataset + '/train_x_' + str(fold) + '.npy')
        train_labels = np.load(FLAGS.data_dir + FLAGS.dataset + '/train_y_' + str(fold) + '.npy')
        validation_images = np.load(FLAGS.data_dir + FLAGS.dataset + '/test_x_' + str(fold) + '.npy')
        validation_labels = np.load(FLAGS.data_dir + FLAGS.dataset + '/test_y_' + str(fold) + '.npy')
        train_num_examples = train_images.shape[0]
        validation_num_examples = validation_images.shape[0]
        print("train size = ", train_num_examples)
        print("validation size = ", validation_num_examples)

        # reduce dataset according to data_fraction parameter
        train_images, _ = util.split_data(train_images, FLAGS.data_fraction)
        train_labels, _ = util.split_data(train_labels, FLAGS.data_fraction)
        validation_images, _ = util.split_data(validation_images, FLAGS.data_fraction)
        validation_labels, _ = util.split_data(validation_labels, FLAGS.data_fraction)

        # set up early stopping
        final_epoch = 0
        final_validation_ce = math.inf
        final_validation_acc = 0
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
                    validation_ce, validation_acc, conf_matrix = session.run([reduce_mean_cross_entropy, reduce_mean_accuracy, confusion_matrix_op], {input: batch_xs, label: batch_ys})
                    ce_vals.append(validation_ce)
                    acc_vals.append(validation_acc)
                    conf_mxs.append(conf_matrix)
                avg_validation_ce = sum(ce_vals) / len(ce_vals)
                avg_validation_acc = sum(acc_vals) / len(acc_vals)
                print('VALIDATION CROSS ENTROPY: ' + str(avg_validation_ce))
                print('VALIDATION ACCURACY: ' + str(avg_validation_acc))
                print('VALIDATION CONFUSION MATRIX:')
                print(str(sum(conf_mxs)))

                # update final results
                if avg_validation_acc > final_validation_acc or patience == 0:
                    final_epoch = epoch
                    final_validation_ce = avg_validation_ce
                    final_validation_acc = avg_validation_acc
                    wait = 0
                    if FLAGS.output_model:
                        path_prefix = saver.save(session, os.path.join(FLAGS.save_dir, modelFile), global_step=0)
                else:
                    wait += 1
                    if wait == patience:
                        break

                # output tensorboard data
                if FLAGS.output_model and epoch==0:
                    file_writer = tf.summary.FileWriter(FLAGS.save_dir, session.graph)


            print("Results for fold", fold)
            print('Final Epoch: ' + str(final_epoch))
            print('Final VALIDATION CROSS ENTROPY: ' + str(final_validation_ce))
            print('Final VALIDATION ACCURACY: ' + str(final_validation_acc))
            k_fold_accuracy.append(final_validation_acc)

    # report average final accuracy across all k folds
    print('Average accuracy across k folds: '+str(sum(k_fold_accuracy) / len(k_fold_accuracy)))


if __name__ == "__main__":
    tf.app.run()
