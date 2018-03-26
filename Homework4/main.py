#---------------------------------------------------
# Daniel Geschwender, Yanfeng Liu, Jeevan Rajagopal
# CSCE 496/896: Deep Learning
# Homework 4
#---------------------------------------------------
# Based off code provided in Hackathon 3, 8, 9

import tensorflow as tf
import numpy as np
import math
import os
import util
import model
from util import print_file

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'directory where data is located')
flags.DEFINE_string('save_dir', 'model', 'directory where model graph and weights are saved')
flags.DEFINE_string('dataset', '', 'dataset to run on')
flags.DEFINE_integer('batch_size', 20, '')
flags.DEFINE_integer('time_scale', 20, '')
flags.DEFINE_integer('max_epoch_num', 50, '')
flags.DEFINE_integer('patience', 0, '')
flags.DEFINE_string('nodes', "100", '')
flags.DEFINE_float('learning_rate', 0.001, '')
flags.DEFINE_float('dropout_rate', 0.0, '')
flags.DEFINE_float('l2_regularizer', 0.0, '')
flags.DEFINE_bool('output_model', False, '')
flags.DEFINE_bool('normalize', False, '')
flags.DEFINE_bool('float16', False, '')
flags.DEFINE_integer('trivial', -1, '')
flags.DEFINE_float('data_fraction', 1.0, '')
flags.DEFINE_integer('samples', 30, '')
flags.DEFINE_integer('seed', 1, '')
flags.DEFINE_integer('fold', 1, '')
flags.DEFINE_string('glove', '', 'glove embedding file')
FLAGS = flags.FLAGS

def main(argv):
    #set up output file
    if not os.path.exists(FLAGS.save_dir):
        os.makedirs(FLAGS.save_dir)
    out = open(FLAGS.save_dir+'/output.txt', 'w')
    
    print_file(tf.__version__, file=out)
    
    embedding, word_map, indexed_words = util.load_glove(FLAGS.glove)
    
    king = util.get_word_vector(embedding,word_map,"king")
    queen = util.get_word_vector(embedding,word_map,"queen")
    man = util.get_word_vector(embedding,word_map,"man")
    woman = util.get_word_vector(embedding,word_map,"woman")
    good = util.get_word_vector(embedding,word_map,"good")
    evil = util.get_word_vector(embedding,word_map,"evil")
    
    print(np.dot(queen, king))
    print(np.dot(queen, king-man+woman))
    print(np.dot(queen, king-woman+man))
    print(np.dot(good, evil))
    
    word = util.get_closest_word(embedding,indexed_words,good+evil)
    print(word)
    word = util.get_closest_word(embedding,indexed_words,-evil)
    print(word)
    word = util.get_closest_word(embedding,indexed_words,-good)
    print(word)
    word = util.get_closest_word(embedding,indexed_words,-king)
    print(word)
    word = util.get_closest_word(embedding,indexed_words,-woman)
    print(word)
    
    # handle command line arguments
    print_file("normalize:" + str(FLAGS.normalize), file=out)
    print_file("regularizer:"+str(FLAGS.l2_regularizer), file=out)
    print_file("dropout_rate:"+str(FLAGS.dropout_rate), file=out)
    print_file("learning_rate:"+str(FLAGS.learning_rate), file=out)
    print_file("nodes:"+str(FLAGS.nodes), file=out)
    print_file("float16:" + str(FLAGS.float16), file=out)
    learning_rate = FLAGS.learning_rate
    dropout_rate = FLAGS.dropout_rate
    nodes = list(map(int, FLAGS.nodes.split(","))) if FLAGS.nodes != "" else []
    regularizer = tf.contrib.layers.l2_regularizer(scale=1.)
    normalize = FLAGS.normalize
    data_type = tf.float16 if FLAGS.float16 else tf.float32
    folds = range(1,5) if FLAGS.fold == 0 else [FLAGS.fold]
    seed = None if FLAGS.seed == 0 else FLAGS.seed
    batch_size = FLAGS.batch_size
    time_scale = FLAGS.time_scale
    np.random.seed(seed)

    # set up the input
    train_num_examples = 100
    train = np.random.randint(embedding.shape[0], size=(train_num_examples, 21))
    input_data = tf.placeholder(tf.int32, [None, 20], name='input_data')
    target_data = tf.placeholder(tf.int32, [None, 20], name='target_data')
    embedding_tensor = tf.constant(embedding,dtype=tf.float32)
    embedding_input = tf.nn.embedding_lookup(embedding_tensor, input_data)
    embedding_target = tf.nn.embedding_lookup(embedding_tensor, target_data)
    
    print(train)
    print(train.shape)
    print(embedding_input)
    print(embedding_target)
    
    # specify the network
    output = model.lstm(embedding_input, nodes, batch_size)
    
    print(output)

    # define classification loss
    loss = tf.losses.cosine_distance(embedding_target, output, dim=2)
    total_loss = loss
    
    # set up training and saving functionality
    global_step_tensor = tf.get_variable('global_step', trainable=False, shape=[], initializer=tf.zeros_initializer)
    optimizer = tf.train.RMSPropOptimizer(learning_rate = learning_rate)
    train_op = optimizer.minimize(total_loss, global_step=global_step_tensor)
    saver = tf.train.Saver()

    k_fold_psnr = []

    for fold in folds:
        print_file("Beginning fold "+str(fold), file=out)
        
        # set up early stopping
        final_epoch = 0
        final_validation_psnr = 0
        patience = FLAGS.patience
        wait = 0

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            
            # run training
            for epoch in range(FLAGS.max_epoch_num):
                print_file('Epoch: ' + str(epoch), file=out)

                # run gradient steps and report mean loss on train data
                psnr_vals = []
                for i in range(train_num_examples // batch_size):
                    batch_xs = train[i*batch_size:(i+1)*batch_size, :-1]
                    batch_ys = train[i*batch_size:(i+1)*batch_size, 1:]
                    _ = session.run(train_op, {input_data: batch_xs, target_data: batch_ys})
                avg_train_psnr = sum(psnr_vals) / len(psnr_vals)
                print_file('TRAIN PSNR: ' + str(avg_train_psnr), file=out)

                # report mean validation loss
                psnr_vals = []
                for i in range(validation_num_examples // batch_size):
                    batch_xs = validation_images[i*batch_size:(i+1)*batch_size, :, :, :]
                    validation_psnr = session.run(mean_psnr, {encoder_input: batch_xs})
                    psnr_vals.append(validation_psnr)
                avg_validation_psnr = sum(psnr_vals) / len(psnr_vals)
                print_file('VALIDATION PSNR: ' + str(avg_validation_psnr), file=out)
                
                # record image for tracking progress
                sample_in = validation_images[0:1, :, :, :]
                sample_out = session.run(tf.cast(decoder_output,tf.uint8), {encoder_input: sample_in})
                imageProgress = np.append(imageProgress, sample_out[0:1, :, :, :], axis=0)

                # update final results
                if avg_validation_psnr > final_validation_psnr or patience == 0:
                    final_epoch = epoch
                    final_validation_psnr = avg_validation_psnr
                    wait = 0
                    if FLAGS.output_model:
                        path_prefix = saver.save(session, os.path.join(FLAGS.save_dir, modelFile1), global_step=0)
                        path_prefix = saver.save(session, os.path.join(FLAGS.save_dir, modelFile2), global_step=0)
                        path_prefix = saver.save(session, os.path.join(FLAGS.save_dir, modelFile3), global_step=0)
                        path_prefix = saver.save(session, os.path.join(FLAGS.save_dir, modelFile4), global_step=0)
                else:
                    wait += 1
                    if wait == patience:
                        break

                # output tensorboard data
                if FLAGS.output_model and epoch==0:
                    file_writer = tf.summary.FileWriter(FLAGS.save_dir, session.graph)

            print_file("Results for fold"+str(fold), file=out)
            print_file('Final Epoch: ' + str(final_epoch), file=out)
            print_file('Final VALIDATION PSNR: ' + str(final_validation_psnr), file=out)
            k_fold_psnr.append(final_validation_psnr)
            
            #sample images
            sample = np.random.permutation(validation_num_examples)
            sample_in = validation_images[sample[:FLAGS.samples] , :, :, :]
            sample_out = session.run(tf.cast(decoder_output,tf.uint8), {encoder_input: sample_in})
            np.save(FLAGS.save_dir+"/sampleIn",sample_in)
            np.save(FLAGS.save_dir+"/sampleOut",sample_out)
            
            # save image progress
            np.save(FLAGS.save_dir+"/progress",imageProgress)
                
    # report average final psnr across all k folds
    print_file('Average psnr across k folds: '+str(sum(k_fold_psnr) / len(k_fold_psnr)), file=out)


if __name__ == "__main__":
    tf.app.run()
