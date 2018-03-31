import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import logging
from time import gmtime, strftime
import os

logging.basicConfig(level = logging.DEBUG)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("mode", "train", "train, eval, or predict")
tf.app.flags.DEFINE_integer("steps", 200000, "number of steps")
tf.app.flags.DEFINE_integer("steps_per_test", 1000, "number of steps per test")
tf.app.flags.DEFINE_integer("batch_size", 64, "number of data per step")
tf.app.flags.DEFINE_float("learning_rate", 0.01, "learning rate")
tf.app.flags.DEFINE_string("ckpt", None, "previous model and parameter")

def logits_fn(x):
    """
    y = wx + b

    placeholder x:shape(None, 784)

    """

    w = tf.Variable(tf.truncated_normal([784, 10], mean = 0, stddev = 1.0, name='random_init'), name='w')
    b = tf.Variable(tf.truncated_normal([10], mean = 0, stddev = 1.0, name='random_init'), name='b')

    return tf.matmul(x, w) + b

def write_accuracy_file_by_list(value_list, dest_folder):
    with open(os.path.join(dest_folder, "accuracy.txt"), mode='w') as f:
        for write_line in value_list:
            f.write("%s\n" % (write_line))

def _load_graph(ckpt):
    meta_file = ckpt + ".meta"
    tf.train.import_meta_graph(meta_file)


def _load_parameter(sess, ckpt):
    saver = tf.train.Saver()
    if os.path.isdir(ckpt):
        ckpt = tf.train.latest_checkpoint(ckpt)
    saver.restore(sess, ckpt)

def main(unused):


    x = tf.placeholder(tf.float32, [None, 784], name='input_data')
    label = tf.placeholder(tf.int32, [None, 10], name='input_label')

    with tf.name_scope('logits'):
        logit = logits_fn(x)

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels= label, logits=logit))

    #In order to save global step, we need to declare it as a variable and pass it into optimizer. Optimizer will help to increase it when updating parameters.
    global_step = tf.Variable(0, name='global_step', trainable=False)

    with tf.name_scope('optimizer'):
        train = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(loss, global_step=global_step)

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

        model_result = os.path.join(default_dir, 'model_result')
        if not (os.path.isdir(model_result)):
            os.makedirs(model_result)

        checkpoint_prefix = os.path.join(default_dir, 'parameters')

        return summary_path, model_result, checkpoint_prefix

    summary_path, model_result, checkpoint_prefix = prepareResultFolder()

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        if FLAGS.ckpt != None:
            _load_parameter(session, FLAGS.ckpt)

        summary_writer = tf.summary.FileWriter(summary_path, session.graph)

        summary_train_op = tf.summary.merge_all(key='train')

        best_train_accuracy = 0
        best_train_step = 0

        best_test_accuracy = 0
        best_test_step = 0

        for _ in range(FLAGS.steps):
            step = global_step.eval()
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
                    tf.train.Saver().save(session, save_path=checkpoint_prefix, global_step=step)

            summary_writer.flush()

        result_train_description = "best_train_accuracy %s, best_train_step %s" % (best_train_accuracy, best_train_step)
        result_test_description ="best_test_accuracy %s, best_test_step %s" % (best_test_accuracy, best_test_step)

        logging.debug(result_train_description)
        logging.debug(result_test_description)
        write_accuracy_file_by_list(value_list=[result_train_description, result_test_description], dest_folder=model_result)

if __name__ == '__main__':
    tf.app.run()

