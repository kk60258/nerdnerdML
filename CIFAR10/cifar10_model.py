import re

import tensorflow as tf

from cifar10_const import *


def inference(images):
    """Build the CIFAR-10 model.

    Args:
      images: Images returned from distorted_inputs() or inputs().

    Returns:
      Logits.
    """

    # layer1
    with tf.variable_scope('layer1') as scope:
        conv1 = tf.layers.conv2d(inputs=images, filters=64, kernel_size=[5, 5], strides=(1, 1), padding='same', activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1), name='conv')
        pool1 = tf.layers.max_pooling2d(conv1, pool_size=[3, 3], strides=[2,2], padding='same', name='pool')
        batch1 = tf.layers.batch_normalization(inputs=pool1, name='batch')
        layer1 = batch1


    # layer2
    with tf.variable_scope('layer2') as scope:
        conv2 = tf.layers.conv2d(inputs=layer1, filters=80, kernel_size=[5, 5], strides=(1, 1), padding='same', activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1), name='conv')
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3, 3], strides=[2,2], padding='same', name='pool')
        batch2 = tf.layers.batch_normalization(inputs=pool2, name='batch')
        layer2 = batch2

    # layer3
    with tf.variable_scope('layer3') as scope:
        conv3 = tf.layers.conv2d(inputs=layer2, filters=128, kernel_size=[3, 3], strides=(1, 1), padding='same', activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1), name='conv')
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=[2,2], padding='same', name='pool')
        batch3 = tf.layers.batch_normalization(inputs=pool3, name='batch')
        layer3 = batch3

    # layer4
    with tf.variable_scope('layer4') as scope:
        flatten = tf.layers.flatten(inputs=layer3, name='flatten')
        dense1 = tf.layers.dense(inputs=flatten, units=100, activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1), name='dense1')
        dense2 = tf.layers.dense(inputs=dense1, units=10, activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1), name='dense2')
        layer4 = dense2
    # We don't apply softmax here because
    # and performs the softmax internally for efficiency.


    return layer4


def loss(logits, labels):
    """Add L2Loss to all the trainable variables.

    Add summary for "Loss" and "Loss/avg".
    Args:
      logits: Logits from inference().
      labels: Labels from distorted_inputs or inputs(). 1-D tensor
              of shape [batch_size]

    Returns:
      Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1)), tf.float32))

    tf.summary.scalar('train accuracy', accuracy, collections=['train'])
    tf.summary.scalar('test accuracy', accuracy, collections=['test'])

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    l2_loss = tf.losses.get_regularization_loss()
    return cross_entropy_mean + l2_loss, accuracy



def train(total_loss, global_step):
    """Train CIFAR-10 model.

    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.

    Args:
      total_loss: Total loss from loss().
      global_step: Integer Variable counting the number of training steps
        processed.
    Returns:
      train_op: op for training.
    """
    # Variables that affect learning rate.
    # num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / batch_size
    # decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # # Decay the learning rate exponentially based on the number of steps.
    # lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
    #                                 global_step,
    #                                 decay_steps,
    #                                 LEARNING_RATE_DECAY_FACTOR,
    #                                 staircase=True)
    # tf.summary.scalar('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(total_loss, global_step=global_step)

    tf.summary.scalar('total loss', total_loss, collections=['train'])
    tf.summary.scalar('total test loss', total_loss, collections=['test'])
    tf.summary.scalar('learning rate', optimizer._lr, collections=['train'])

    return train_op


def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.

    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable

    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.

    Returns:
      Variable Tensor
    """
    dtype = tf.float32

    var = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.

    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % 'tower', '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity',
                      tf.nn.zero_fraction(x))