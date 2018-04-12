#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

# import the inspect_checkpoint library
from tensorflow.python.tools import inspect_checkpoint as chkp


import tensorflow as tf

#ensure that all elements are printed
import numpy
numpy.set_printoptions(threshold=numpy.nan)

FLAGS = None


def main(_):
  tf.reset_default_graph()
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.get_variable("W",shape = [784, 10], initializer = tf.zeros_initializer);
  #W = tf.Variable(tf.zeros([784, 10]))
  b = tf.get_variable("b",shape = [10], initializer = tf.zeros_initializer);
  #b = tf.Variable("v2",tf.zeros([10]))
  y = tf.matmul(x, W) + b
  # use softmax will be 0.8 accuracy
  #y = tf.nn.softmax(tf.matmul(x, W) + b)

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  # 0.5 -> 0.05
  train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

  sess = tf.InteractiveSession()

  
  
  #accuracy function
  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', accuracy)

  # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
  merged = tf.summary.merge_all()
  
  #write log file for tensorBoard
  writer = tf.summary.FileWriter("/tmp/shun_graph/mnist_softmax/", sess.graph)
  
  tf.global_variables_initializer().run()
  
  accuracy_list = ["summary:"]
  # Train
  for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    
    if (i%100) == 1:
        summary, acc = sess.run([merged, accuracy], feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels})
        writer.add_summary(summary,i)
        accuracy_list.append("[train] accuracy after %d step: %f" % (i,acc))

  writer.close()

  # Save the variables to disk.
  save_path = tf.train.Saver().save(sess, "/tmp/shun_model.ckpt")
  print("Model saved in path: %s" % save_path)
  
  # print all tensors in checkpoint file
  #chkp.print_tensors_in_checkpoint_file("/tmp/shun_model.ckpt", tensor_name='', all_tensors=True)
  
  
  #summary
  accuracy_list.append("[final] accuracy after 1000 step: %f" % sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels}))
  for list in accuracy_list:
      print(list)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)