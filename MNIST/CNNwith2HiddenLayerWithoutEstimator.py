from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from time import gmtime, strftime
import os
import logging

logging.basicConfig(level = logging.DEBUG)

# mnist_images = input_data.read_data_sets("MNIST_data/", one_hot=True)
# total_size = mnist_images.train.num_examples
# batch_size = 1000
# epoch = 100
# display_epoch = 2
# learning_rate = 0.005
# n_classes = 10

FLAGS = tf.app.flags.FLAGS
# tf.app.flags.DEFINE_string("param_name", "default_val", "description")
tf.app.flags.DEFINE_string("mode", "train", "train, eval, or predict")
tf.app.flags.DEFINE_integer("steps", 1, "number of steps")
tf.app.flags.DEFINE_integer("batch_size", 64, "number of data per step")
tf.app.flags.DEFINE_float("learning_rate", 0.01, "learning rate")


tf.logging.set_verbosity(tf.logging.DEBUG)


def cnn_model_fn(features, labels, mode):

    #input shape = [size, 28, 28, 1]
    input_layer = tf.reshape(features, [-1, 28, 28, 1])

    # Convolutional Layer #1 . out size*32*28*28
    conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    # Pooling Layer #1 . out size*32*14*14
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 . out size*64*14*14
    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    # Pooling Layer #2 . out size*64*7*7
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    # dropout
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == 'predict':
        return predictions

    # Calculate Loss (for both TRAIN and EVAL modes)
    # onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=tf.argmax(input=labels, axis=1), predictions=predictions["classes"])}


    return train_op, eval_metric_ops


def main(unused_argv):
    batch_size = FLAGS.batch_size
    steps = FLAGS.steps
    learning_rate = FLAGS.learning_rate

    def prepareResultFolder():
        timestring = strftime("%Y_%b_%d_%H_%M_%S", gmtime())
        default_dir = os.path.join(os.getcwd(), timestring, "{}-step_{}-learningrate_{}-batchsize".format(steps, learning_rate, batch_size))
        summary_path = os.path.join(default_dir, 'summary_dir')
        if not (os.path.isdir(summary_path)):
            os.makedirs(summary_path)

        model_path = os.path.join(default_dir, 'model_dir')
        if not (os.path.isdir(model_path)):
            os.makedirs(model_path)

        return summary_path, model_path

    summary_path, model_path = prepareResultFolder()

    features = tf.placeholder(tf.float32, [None, 784])
    labels = tf.placeholder(tf.float32, [None, 10])

    if FLAGS.mode == 'train' :
        train_op, eval_op = cnn_model_fn(features, labels, FLAGS.mode)
        # train_summary = tf.summary.scalar('train loss', train_op)
        # eval_summary = tf.summary.scalar('eval', eval_op)

    # prepare data
    mnist_images = input_data.read_data_sets("MNIST_data/", one_hot=True)


    with tf.Session() as session:

        summary_writer = tf.summary.FileWriter(summary_path, session.graph)

        session.run(tf.global_variables_initializer())
        session.run(tf.local_variables_initializer())

        tf.summary.scalar('accuracy', eval_op["accuracy"][0])
        merged = tf.summary.merge_all()

        logging.debug('12312312312312312')
        for step in range(steps):
            train_data, train_label = mnist_images.train.next_batch(batch_size)
            _, train_acc, train_summary = session.run([train_op, eval_op["accuracy"], merged], feed_dict={features: train_data, labels: train_label})
            summary_writer.add_summary(train_summary, step)
            # summary_writer.add_summary(losses, step)

        test_data, test_label = mnist_images.test.images, mnist_images.test.labels
        evals = session.run(eval_op["accuracy"], feed_dict={features: test_data, labels: test_label})
        logging.debug('eval op %s', evals)
        # summary_writer.add_summary(summary_result, step)
            # save_path = tf.train.Saver().save(session, save_path=model_path, global_step=step)



if __name__ == '__main__':
    tf.app.run()






