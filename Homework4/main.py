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
flags.DEFINE_integer('batch_size', 10, '')
flags.DEFINE_integer('time_scale', 20, '')
flags.DEFINE_integer('max_epoch_num', 50, '')
flags.DEFINE_integer('patience', 0, '')
flags.DEFINE_string('nodes', "100", '')
flags.DEFINE_float('learning_rate', 0.001, '')
flags.DEFINE_float('dropout_rate', 0.0, '')
flags.DEFINE_float('l2_regularizer', 0.0, '')
flags.DEFINE_bool('output_model', False, '')
flags.DEFINE_integer('trivial', -1, '')
flags.DEFINE_float('data_fraction', 1.0, '')
flags.DEFINE_integer('samples', 30, '')
flags.DEFINE_integer('seed', 1, '')
flags.DEFINE_integer('fold', 1, '')
flags.DEFINE_integer('sentence_length', 10, '')
flags.DEFINE_integer('prediction_length', 1, '')
flags.DEFINE_string('glove', '', 'glove embedding file')
flags.DEFINE_string('text_data', '/work/cse496dl/shared/hackathon/09/train.en', 'text data file')
flags.DEFINE_bool('load_data', False, '')
FLAGS = flags.FLAGS

def main(argv):
    #set up output file
    if not os.path.exists(FLAGS.save_dir):
        os.makedirs(FLAGS.save_dir)
    out = open(FLAGS.save_dir+'/output.txt', 'w')
    
    print_file(tf.__version__, file=out)
        
    # handle command line arguments
    print_file("sentence_length:"+str(FLAGS.sentence_length), file=out)
    print_file("prediction_length:"+str(FLAGS.prediction_length), file=out)
    print_file("regularizer:"+str(FLAGS.l2_regularizer), file=out)
    print_file("dropout_rate:"+str(FLAGS.dropout_rate), file=out)
    print_file("learning_rate:"+str(FLAGS.learning_rate), file=out)
    print_file("nodes:"+str(FLAGS.nodes), file=out)
    print_file("glove:"+str(FLAGS.glove), file=out)
    sentence_length = FLAGS.sentence_length
    prediction_length = FLAGS.prediction_length
    learning_rate = FLAGS.learning_rate
    dropout_rate = FLAGS.dropout_rate
    nodes = list(map(int, FLAGS.nodes.split(","))) if FLAGS.nodes != "" else []
    regularizer = tf.contrib.layers.l2_regularizer(scale=1.)
    folds = range(1,5) if FLAGS.fold == 0 else [FLAGS.fold]
    seed = None if FLAGS.seed == 0 else FLAGS.seed
    batch_size = FLAGS.batch_size
    time_scale = FLAGS.time_scale
    np.random.seed(seed)

    # set up the input
    embedding, word_map, indexed_words = util.load_glove(FLAGS.glove)
    input_data = tf.placeholder(tf.int32, [None, sentence_length-prediction_length], name='input_data')
    decoder_data = tf.placeholder(tf.int32, [None, prediction_length], name='decoder_data')
    target_data = tf.placeholder(tf.int32, [None, prediction_length], name='target_data')
    embedding_tensor = tf.constant(embedding,dtype=tf.float32)
    embedding_input = tf.nn.embedding_lookup(embedding_tensor, input_data)
    embedding_decoder = tf.nn.embedding_lookup(embedding_tensor, decoder_data)
    embedding_target = tf.nn.embedding_lookup(embedding_tensor, target_data)
    
    # load source vocab and data
    if FLAGS.load_data:
        util.setup_data(FLAGS.text_data, word_map, sentence_length)
    data = np.load("textData.npy")
    
    # set up the training/validation data
    #data = np.random.randint(embedding.shape[0], size=(200, sentence_length))
    train, validation = util.split_data(data, 0.9)
    train_num_examples = train.shape[0]
    validation_num_examples = validation.shape[0]
    print(train_num_examples, validation_num_examples)
    
    # specify the network
    output = model.sentence_completion_rnn(embedding_input, embedding_decoder, nodes, batch_size)
    
    print(output)

    # define classification loss
    element_distances = embedding_target*output
    word_distances = tf.reduce_sum(element_distances, [2])
    sentence_distances = tf.reduce_mean(word_distances, [1])
    mean_distances = tf.reduce_mean(sentence_distances)
    print(element_distances)
    print(sentence_distances)
    total_loss = tf.square((-sentence_distances+1)*10)
    
    # set up training and saving functionality
    global_step_tensor = tf.get_variable('global_step', trainable=False, shape=[], initializer=tf.zeros_initializer)
    optimizer = tf.train.RMSPropOptimizer(learning_rate = learning_rate)
    train_op = optimizer.minimize(total_loss, global_step=global_step_tensor)
    saver = tf.train.Saver()

    k_fold_distance = []

    for fold in folds:
        print_file("Beginning fold "+str(fold), file=out)
        
        # set up early stopping
        final_epoch = 0
        final_validation_distance = 0
        patience = FLAGS.patience
        wait = 0

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            
            # run training
            for epoch in range(FLAGS.max_epoch_num):
                print_file('Epoch: ' + str(epoch), file=out)

                # run gradient steps and report mean loss on train data
                distance_vals = []
                for i in range(train_num_examples // batch_size):
                    batch_xs = train[i*batch_size:(i+1)*batch_size, :-prediction_length]
                    batch_ys = train[i*batch_size:(i+1)*batch_size, -prediction_length:]
                    _,train_distance = session.run([train_op, mean_distances], {input_data: batch_xs, decoder_data: np.zeros((batch_size, prediction_length)), target_data: batch_ys})
                    distance_vals.append(train_distance)
                avg_train_distance = sum(distance_vals) / len(distance_vals)
                print_file('TRAIN DISTANCE: ' + str(avg_train_distance), file=out)

                # report mean validation loss
                distance_vals = []
                for i in range(validation_num_examples // batch_size):
                    batch_xs = validation[i*batch_size:(i+1)*batch_size, :-prediction_length]
                    batch_ys = validation[i*batch_size:(i+1)*batch_size, -prediction_length:]
                    validation_distance = session.run(mean_distances, {input_data: batch_xs, decoder_data: np.zeros((batch_size, prediction_length)), target_data: batch_ys})
                    distance_vals.append(validation_distance)
                avg_validation_distance = sum(distance_vals) / len(distance_vals)
                print_file('VALIDATION DISTANCE: ' + str(avg_validation_distance), file=out)
                
                # output example sentence
                a,b,c,d = session.run([embedding_input, embedding_target, output, word_distances], {input_data: validation[0:1, :-prediction_length], decoder_data: np.zeros((batch_size, prediction_length)), target_data: validation[0:1,-prediction_length:]})
                print_file("EXAMPLE INPUT: "+util.get_sentence(embedding, indexed_words, a[0]), file=out)
                print_file("EXAMPLE TARGET: "+util.get_sentence(embedding, indexed_words, b[0]), file=out)
                print_file("EXAMPLE OUTPUT: "+util.get_sentence(embedding, indexed_words, c[0]), file=out)
                print_file("EXAMPLE SIMILARITY: "+str(d[0]), file=out)

                # update final results
                if avg_validation_distance > final_validation_distance or patience == 0:
                    final_epoch = epoch
                    final_validation_distance = avg_validation_distance
                    wait = 0
                else:
                    wait += 1
                    if wait == patience:
                        break

                # output tensorboard data
                if FLAGS.output_model and epoch==0:
                    file_writer = tf.summary.FileWriter(FLAGS.save_dir, session.graph)

            print_file("Results for fold"+str(fold), file=out)
            print_file('Final Epoch: ' + str(final_epoch), file=out)
            print_file('Final VALIDATION DISTANCE: ' + str(final_validation_distance), file=out)
            k_fold_distance.append(final_validation_distance)
                
    # report average final distance across all k folds
    print_file('Average distance across k folds: '+str(sum(k_fold_distance) / len(k_fold_distance)), file=out)


if __name__ == "__main__":
    tf.app.run()
