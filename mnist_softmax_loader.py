#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 11:06:42 2018

@author: shun
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

# import the inspect_checkpoint library
from tensorflow.python.tools import inspect_checkpoint as chkp

#ensure that all elements are printed
import numpy
numpy.set_printoptions(threshold=numpy.nan)

import tensorflow as tf

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


  sess = tf.InteractiveSession()
  
  # load saved model
  tf.train.Saver().restore(sess, "/tmp/shun_model.ckpt")
  
  # print all tensors in checkpoint file
  chkp.print_tensors_in_checkpoint_file("/tmp/shun_model.ckpt", tensor_name='', all_tensors=True)
  

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)