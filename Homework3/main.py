#---------------------------------------------------
# Daniel Geschwender, Yanfeng Liu, Jeevan Rajagopal
# CSCE 496/896: Deep Learning
# Homework 3
#---------------------------------------------------
# Based off code provided in Hackathon 3

import tensorflow as tf
import numpy as np
import math
import os
import util
import model

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'directory where data is located')
flags.DEFINE_string('save_dir', 'model', 'directory where model graph and weights are saved')
flags.DEFINE_string('dataset', '', 'dataset to run on')
flags.DEFINE_integer('batch_size', 32, '')
flags.DEFINE_integer('max_epoch_num', 50, '')
flags.DEFINE_integer('patience', 0, '')
flags.DEFINE_string('filters', "2,0,4,0", '')
flags.DEFINE_integer('code_size', "100", '')
flags.DEFINE_string('linear_nodes', "", '')
flags.DEFINE_float('learning_rate', 0.001, '')
flags.DEFINE_float('dropout_rate', 0.0, '')
flags.DEFINE_float('l2_regularizer', 0.0, '')
flags.DEFINE_bool('output_model', False, '')
flags.DEFINE_float('data_fraction', 1.0, '')
flags.DEFINE_integer('samples', 30, '')
flags.DEFINE_integer('seed', 0, '')
flags.DEFINE_integer('fold', 1, '')
FLAGS = flags.FLAGS

def main(argv):
    #set up output file
    if not os.path.exists(FLAGS.save_dir):
        os.makedirs(FLAGS.save_dir)
    out = open(FLAGS.save_dir+'/output.txt', 'w')
    
    print(tf.__version__, file=out)

    # handle command line arguments
    print("regularizer:",FLAGS.l2_regularizer, file=out)
    print("dropout_rate:",FLAGS.dropout_rate, file=out)
    print("learning_rate:",FLAGS.learning_rate, file=out)
    print("filters:",FLAGS.filters, file=out)
    print("code_size:",FLAGS.code_size, file=out)
    learning_rate = FLAGS.learning_rate
    dropout_rate = FLAGS.dropout_rate
    filters = list(map(int, FLAGS.filters.split(","))) if FLAGS.filters != "" else []
    linear_nodes = list(map(int, FLAGS.linear_nodes.split(","))) if FLAGS.linear_nodes != "" else []
    regularizer = tf.contrib.layers.l2_regularizer(scale=1.)
    code_size = FLAGS.code_size
    folds = range(1,5) if FLAGS.fold == 0 else [FLAGS.fold]
    seed = None if FLAGS.seed == 0 else FLAGS.seed
    modelFile = "maxquality_encoder_homework_3"
    np.random.seed(seed)

    # specify the network
    encoder_input, encoder_output, encoder_training, precode_shape = model.encoder(filters=filters, code_size=code_size, regularizer=regularizer, dropout_rate=dropout_rate)
    decoder_input, decoder_output, decoder_training = model.decoder(encoder_output, filters=filters, postcode_shape=precode_shape, regularizer=regularizer, dropout_rate=dropout_rate)

    # define classification loss
    psnr  = model.psnr(encoder_input, decoder_output)
    mean_psnr  = tf.reduce_mean(psnr)
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    REG_COEFF = FLAGS.l2_regularizer
    total_loss = -mean_psnr + REG_COEFF * sum(regularization_losses)
    

    # set up training and saving functionality
    global_step_tensor = tf.get_variable('global_step', trainable=False, shape=[], initializer=tf.zeros_initializer)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    train_op = optimizer.minimize(total_loss, global_step=global_step_tensor)
    saver = tf.train.Saver()

    k_fold_psnr = []

    for fold in folds:
        print("Beginning fold ", fold, file=out)

        # load data
        (train_images, train_labels), (validation_images, validation_labels) = tf.keras.datasets.cifar100.load_data()

        # reduce dataset according to data_fraction parameter
        train_images, _ = util.split_data(train_images, FLAGS.data_fraction)
        train_labels, _ = util.split_data(train_labels, FLAGS.data_fraction)
        validation_images, _ = util.split_data(validation_images, FLAGS.data_fraction)
        validation_labels, _ = util.split_data(validation_labels, FLAGS.data_fraction)

        train_num_examples = train_images.shape[0]
        validation_num_examples = validation_images.shape[0]
        print("train size = ", train_num_examples, file=out)
        print("validation size = ", validation_num_examples, file=out)
        
        # set up early stopping
        final_epoch = 0
        final_validation_psnr = 0
        patience = FLAGS.patience
        wait = 0

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())

            # run training
            batch_size = FLAGS.batch_size
            for epoch in range(FLAGS.max_epoch_num):
                print('Epoch: ' + str(epoch), file=out)

                # run gradient steps and report mean loss on train data
                psnr_vals = []
                for i in range(train_num_examples // batch_size):
                    batch_xs = train_images[i*batch_size:(i+1)*batch_size, :, :, :]
                    _, train_psnr = session.run([train_op, mean_psnr], {encoder_input: batch_xs, encoder_training: True, decoder_training: True})
                    psnr_vals.append(train_psnr)
                avg_train_psnr = sum(psnr_vals) / len(psnr_vals)
                print('TRAIN PSNR: ' + str(avg_train_psnr), file=out)

                # report mean validation loss
                psnr_vals = []
                for i in range(validation_num_examples // batch_size):
                    batch_xs = validation_images[i*batch_size:(i+1)*batch_size, :, :, :]
                    validation_psnr = session.run(mean_psnr, {encoder_input: batch_xs})
                    psnr_vals.append(validation_psnr)
                avg_validation_psnr = sum(psnr_vals) / len(psnr_vals)
                print('VALIDATION PSNR: ' + str(avg_validation_psnr), file=out)

                # update final results
                if avg_validation_psnr > final_validation_psnr or patience == 0:
                    final_epoch = epoch
                    final_validation_psnr = avg_validation_psnr
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

            print("Results for fold", fold, file=out)
            print('Final Epoch: ' + str(final_epoch), file=out)
            print('Final VALIDATION PSNR: ' + str(final_validation_psnr), file=out)
            k_fold_psnr.append(final_validation_psnr)
            
            #sample images
            sample = np.random.permutation(validation_num_examples)
            sample_in = validation_images[sample[:FLAGS.samples] , :, :, :]
            sample_out = session.run(tf.cast(decoder_output,tf.uint8), {encoder_input: sample_in})
            np.save(FLAGS.save_dir+"/sampleIn",sample_in)
            np.save(FLAGS.save_dir+"/sampleOut",sample_out)
                
    # report average final psnr across all k folds
    print('Average psnr across k folds: '+str(sum(k_fold_psnr) / len(k_fold_psnr)), file=out)


if __name__ == "__main__":
    tf.app.run()
