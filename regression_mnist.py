# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 22:43:07 2018

@author: abra
"""

from __future__ import absolute_import
from __future__ import division
from datetime import timedelta
from tensorflow.examples.tutorials.mnist import input_data

import numpy as np
import tensorflow as tf
import time

tf.logging.set_verbosity(tf.logging.INFO)

num_iterations = 1000
batch_size = 100
required_improvement = 100

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("MODE", "train", "train / predict")

def main(unused_argv):
    mnist = input_data.read_data_sets('./MNIST-data', one_hot=True)
    
    x = tf.placeholder(tf.float32, [None, 784])
    w = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, w) + b
    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy_function = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 
    tf.summary.scalar('accuracy_function', accuracy_function)
    merged = tf.summary.merge_all()
    
    saver = tf.train.Saver()
    
    if FLAGS.MODE == 'train':
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter('./MNIST-graph', sess.graph)
        start_time = time.time()
        
        total_iterations = 0
        best_validation_accuracy = 0.0
        accuracy = 0.0
        for i in range(num_iterations):
            total_iterations += 1
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
            if (total_iterations % 10 == 0) or (i == (num_iterations - 1)):  
                summary, accuracy = sess.run([merged, accuracy_function], feed_dict={x: mnist.validation.images, y_: mnist.validation.labels})
                train_writer.add_summary(summary, i)
                print('Validation Accuracy of Iteration ' + str(total_iterations) + ': ' + str(accuracy))
                if accuracy > best_validation_accuracy:
                    best_validation_accuracy = accuracy
                    last_improvement = total_iterations
                    saver.save(sess, './MNIST-model/model.ckpt')
                    print(' * better accuracy')
                if (total_iterations - last_improvement > required_improvement):
                    print('No improvement found in ' + str(required_improvement) + ' steps')
                    print('Last iteration: ' + str(last_improvement))
                    break
        
        end_time = time.time()
        print('Spent time: ' + str(timedelta(seconds = int(round(end_time - start_time)))))
        
    
    if FLAGS.MODE == 'predict':
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, './MNIST-model/model.ckpt') 
            accuracy = sess.run(accuracy_function, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
            print('Predicted Test Accuracy :' + str(accuracy))

if __name__ == "__main__":
    tf.app.run()