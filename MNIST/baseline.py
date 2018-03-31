import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import logging
from time import gmtime, strftime
import os

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

    w = tf.Variable(tf.truncated_normal([784, 10], mean = 0, stddev = 1.0, name='random_init'), name='w')
    b = tf.Variable(tf.truncated_normal([10], mean = 0, stddev = 1.0, name='random_init'), name='b')

    return tf.matmul(x, w) + b

def main(unused):


    x = tf.placeholder(tf.float32, [None, 784], name='input_data')
    label = tf.placeholder(tf.int32, [None, 10], name='input_label')

    with tf.name_scope('logits'):
        logit = logits_fn(x)

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels= label, logits=logit))

    with tf.name_scope('optimizer'):
        train = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(loss)

    tf.summary.scalar('loss', loss, collections=['train'], family='train_tab')

    # equal_per_label = tf.equal(tf.arg_max(logit, 1), tf.arg_max(label, 1))
    # accuracy_per_label = tf.cast(equal_per_label, dtype=tf.float32)

    """
    This output a label shape size of int32 array
    predictions: Logits tensor, float - [batch_size, NUM_CLASSES].
    targets: Labels tensor, int32 - [batch_size], with values in the range[0, NUM_CLASSES).
    k: top k tolerance 
    """
    with tf.name_scope('accuracy'):
        correct = tf.cast(tf.nn.in_top_k(logit, tf.arg_max(label, 1), 1), tf.float32)

        correct_count = tf.reduce_sum(correct)
        accuracy = tf.reduce_mean(correct)

    tf.summary.scalar('accuracy', accuracy, collections=['train'], family='train_tab')
    # prepare data
    mnist_images = input_data.read_data_sets("MNIST_data/", one_hot=True)

    def prepareResultFolder():
        timestring = strftime("%Y_%b_%d_%H_%M_%S", gmtime())
        default_dir = os.path.join(os.getcwd(), timestring, "{}-step_{}-learningrate_{}-batchsize".format(FLAGS.steps, FLAGS.learning_rate, FLAGS.batch_size))
        summary_path = os.path.join(default_dir, 'summary_dir')
        if not (os.path.isdir(summary_path)):
            os.makedirs(summary_path)

        model_path = os.path.join(default_dir, 'model_dir')
        if not (os.path.isdir(model_path)):
            os.makedirs(model_path)

        return summary_path, model_path

    summary_path, model_path = prepareResultFolder()

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(summary_path, session.graph)

        summary_train_op = tf.summary.merge_all(key='train')

        best_train_accuracy = 0
        best_train_step = 0

        best_test_accuracy = 0
        best_test_step = 0

        for step in range(FLAGS.steps):
            train_data, train_label = mnist_images.train.next_batch(FLAGS.batch_size)
            _, train_loss, train_accuracy, summary_train = session.run([train, loss, accuracy, summary_train_op], feed_dict={x: train_data, label: train_label})
            logging.debug("step %d, train loss %s, train accuracy %s", step, train_loss, train_accuracy)
            summary_writer.add_summary(summary_train, global_step=step)

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

                summary_test=tf.Summary()
                summary_test.value.add(tag='test accuracy', simple_value = test_accuracy)
                summary_writer.add_summary(summary_test, global_step=step)

                logging.debug("step %d, test accuracy %s", step, test_accuracy)
                if best_test_accuracy < test_accuracy:
                    best_test_accuracy = test_accuracy
                    best_test_step = step

            summary_writer.flush()

        logging.debug("best_train_accuracy %s, best_train_step %s", best_train_accuracy, best_train_step)
        logging.debug("best_test_accuracy %s, best_test_step %s", best_test_accuracy, best_test_step)

if __name__ == '__main__':
    tf.app.run()

