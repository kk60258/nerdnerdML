import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import logging

logging.basicConfig(level = logging.DEBUG)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("mode", "train", "train, eval, or predict")
tf.app.flags.DEFINE_integer("steps", 20000, "number of steps")
tf.app.flags.DEFINE_integer("steps_per_test", 1000, "number of steps per test")
tf.app.flags.DEFINE_integer("batch_size", 64, "number of data per step")
tf.app.flags.DEFINE_float("learning_rate", 0.01, "learning rate")

def logits_fn(x):
    """
    y = wx + b

    placeholder x:shape(None, 784)

    """

    w = tf.Variable(tf.truncated_normal([784, 10], mean = 0, stddev = 1.0))
    b = tf.Variable(tf.truncated_normal([10], mean = 0, stddev = 1.0))

    return tf.matmul(x, w) + b

def main(unused):


    x = tf.placeholder(tf.float32, [None, 784])
    label = tf.placeholder(tf.int32, [None, 10])

    logit = logits_fn(x)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels= label, logits=logit))

    train = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(loss)

    # equal_per_label = tf.equal(tf.arg_max(logit, 1), tf.arg_max(label, 1))
    # accuracy_per_label = tf.cast(equal_per_label, dtype=tf.float32)

    """
    This output a label shape size of int32 array
    predictions: Logits tensor, float - [batch_size, NUM_CLASSES].
    targets: Labels tensor, int32 - [batch_size], with values in the range[0, NUM_CLASSES).
    """
    correct = tf.cast(tf.nn.in_top_k(logit, tf.arg_max(label, 1), 1), tf.float32)

    correct_count = tf.reduce_sum(correct)
    accuracy = tf.reduce_mean(correct)

    # prepare data
    mnist_images = input_data.read_data_sets("MNIST_data/", one_hot=True)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        best_train_accuracy = 0
        best_train_step = 0

        best_test_accuracy = 0
        best_test_step = 0

        for step in range(FLAGS.steps):
            train_data, train_label = mnist_images.train.next_batch(FLAGS.batch_size)
            _, train_loss, train_accuracy = session.run([train, loss, accuracy], feed_dict={x: train_data, label: train_label})
            logging.debug("step %d, train loss %s, train accuracy %s", step, train_loss, train_accuracy)

            if best_train_accuracy < train_accuracy:
                best_train_accuracy = train_accuracy
                best_train_step = step

            if step % FLAGS.steps_per_test == 0 or step + 1 == FLAGS.steps:
                test_epoch_size = mnist_images.test.num_examples
                test_steps = int(test_epoch_size / FLAGS.batch_size)
                test_correct_count = 0

                for test_step in range(test_steps):
                    test_data, test_label = mnist_images.test.next_batch(FLAGS.batch_size)
                    test_correct_count += session.run(correct_count, feed_dict={x: test_data, label: test_label})

                test_accuracy = float(test_correct_count) / (test_steps * FLAGS.batch_size)
                logging.debug("step %d, test accuracy %s", step, test_accuracy)
                if best_test_accuracy < test_accuracy:
                    best_test_accuracy = test_accuracy
                    best_test_step = step

        logging.debug("best_train_accuracy %s, best_train_step %s", best_train_accuracy, best_train_step)
        logging.debug("best_test_accuracy %s, best_test_step %s", best_test_accuracy, best_test_step)

if __name__ == '__main__':
    tf.app.run()

