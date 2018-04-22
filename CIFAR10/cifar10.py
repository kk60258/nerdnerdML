import argparse
import logging
import os

import cifar10_data
import cifar10_model
import tensorflow as tf

from cifar10_const import *

logging.basicConfig(level=logging.DEBUG)

tf.train.latest_checkpoint

def download_and_extract():
    cifar10_data.maybe_download_and_extract(FLAGS.data_dir, DATA_URL)


def gcloud_auth():
    if FLAGS.use_google_cloud and FLAGS.help_to_login_google_cloud:
        import nerdcolab.gcloud as gcloud
        gcloud.gcloud_auth(FLAGS.gcloud_project_name)


def gcloud_save():
    if FLAGS.use_google_cloud:
        import nerdcolab.gcloud as gcloud
        gcloud.save_to_bucket(FLAGS.train_dir, FLAGS.gcloud_bucket_name, FLAGS.gcloud_project_name,
                    step=None,
                    save_events=False,
                    force=FLAGS.overwrite_google_cloud_if_filename_conflicts,
                    save_all_dir=False)


def gcloud_load():
    if FLAGS.use_google_cloud:
        import nerdcolab.gcloud as gcloud
        gcloud.load_from_bucket(FLAGS.zip_file_name, FLAGS.gcloud_bucket_name, FLAGS.restore_dir)

def train():
    """Train CIFAR-10 for a number of steps."""
    gcloud_auth()
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()

        # Get images and labels for CIFAR-10.
        # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
        # GPU and resulting in a slow down.
        with tf.device('/cpu:0'):
            dataset = cifar10_data.distorted_inputs(data_dir=FLAGS.data_dir)
            dataset = dataset.batch(batch_size).repeat()
            iterator = dataset.make_one_shot_iterator()
            next_batch = iterator.get_next()

            test_dataset = cifar10_data.distorted_test_inputs(data_dir=FLAGS.data_dir)
            test_dataset = test_dataset.batch(batch_size)
            test_iterator = test_dataset.make_initializable_iterator()
            test_next_batch = test_iterator.get_next()

            images = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
            labels = tf.placeholder(tf.uint8, shape=[None, 10])

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = cifar10_model.inference(images)

        # Calculate loss.
        loss, accuracy = cifar10_model.loss(logits, labels)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = cifar10_model.train(loss, global_step, FLAGS.learning_rate)

        merged = tf.summary.merge_all(key='train')
        merged_test = tf.summary.merge_all(key='test')

        save_result_filename = os.path.join(FLAGS.train_dir, 'model')

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            best_test_accuracy = 0

            if not os.path.isdir(FLAGS.summary_dir):
                os.makedirs(FLAGS.summary_dir)

            summary_writer = tf.summary.FileWriter(FLAGS.summary_dir, sess.graph)

            #restore path not empty
            if os.path.exists(os.path.dirname(FLAGS.restore_model_prefix)):
                #gcloud_load()
                tf.train.Saver().restore(sess, FLAG.restore_model_prefix)

            for _ in range(max_steps):
                train_images, train_labels = sess.run(next_batch)
                #image_op = tf.summary.image('images', train_images)
                # train_labels = tf.reshape(train_labels, shape=[128])
                # train_labels = tf.one_hot(train_labels, depth=10).eval()
                # print(train_labels.eval())
                # print(train_labels.eval().shape)
                _, loss_value, accuracy_value, summary = sess.run([train_op, loss, accuracy, merged], feed_dict={images: train_images, labels: train_labels})

                step = global_step.eval()

                # logging.debug('step {}, train loss {}, accuracy {}'.format(step, loss_value, accuracy_value))

                summary_writer.add_summary(summary, global_step=step)
                #summary_writer.add_summary(image_summary, global_step=step)

                if step % log_frequency == 0 or step + 1 == max_steps:
                    test_epoch_size = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
                    test_steps = int(test_epoch_size / batch_size)
                    test_correct_count = 0

                    sess.run(test_iterator.initializer)

                    for test_step in range(test_steps):
                        test_images, test_labels = sess.run(test_next_batch)
                        test_accuracy = sess.run(accuracy, feed_dict={images: test_images, labels: test_labels})
                        test_correct_count += (test_accuracy * test_images.shape[0])
                        #logging.debug('test_step {}, test_correct_count {}, test_images.shape {}'.format(test_step, test_correct_count, test_images.shape[0]))

                    test_accuracy = float(test_correct_count) / NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

                    summary_test=tf.Summary()
                    summary_test.value.add(tag='test accuracy', simple_value = test_accuracy)
                    summary_writer.add_summary(summary_test, global_step=step)

                    logging.debug('step {}, train loss {}, train accuracy {}, test accuracy {}, best {}'.format(step, loss_value, accuracy_value, test_accuracy, best_test_accuracy))

                    if best_test_accuracy < test_accuracy:
                        best_test_accuracy = test_accuracy
                        best_test_step = step
                        tf.train.Saver().save(sess, save_path=save_result_filename, global_step=step)
                        gcloud_save()

                summary_writer.flush()


def main(argv=None):  # pylint: disable=unused-argument
    download_and_extract()
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Basic model parameters.

    parser.add_argument('--data_dir', type=str, default='/tmp/cifar10_data',
                        help='Path to the CIFAR-10 data directory.')

    parser.add_argument('--train_dir', type=str, default='/tmp/cifar10_train',
                        help='Path to the CIFAR-10 train result directory.')

    parser.add_argument('--restore_model_prefix', type=str, default='/tmp/cifar10_restore/model-5',
                        help='Path to the CIFAR-10 restore ckpt file name prefix.')

    parser.add_argument('--summary_dir', type=str, default='/tmp/cifar10_summary',
                        help='Path to the CIFAR-10 summary result directory.')

    parser.add_argument('--use_google_cloud', type=bool, default=False,
                        help='save to / load from google cloud storage')

    parser.add_argument('--help_to_login_google_cloud', type=bool, default=False,
                        help='help to log in google cloud storage')

    parser.add_argument('--overwrite_google_cloud_if_filename_conflicts', type=bool, default=False,
                        help='overwrite_google_cloud_if_filename_conflicts')

    parser.add_argument('--gcloud_project_name', type=str, default='ai-model-test',
                        help='the project name in google cloud platform')

    parser.add_argument('--gcloud_bucket_name', type=str, default='ai-model-test-ml',
                        help='the bucket name in google cloud storage')

    parser.add_argument('--zip_file_name', type=str, default='cifar10.zip',
                        help='the zip file name in bucket of google cloud storage to restore from')

    parser.add_argument('--max_steps', type=int, default=3,
                        help='')

    parser.add_argument('--log_frequency', type=int, default=1,
                        help='')

    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='')

    FLAGS = parser.parse_args()
    batch_size = 128

    global max_steps, log_frequency
    max_steps = FLAGS.max_steps
    log_frequency = FLAGS.log_frequency

    tf.app.run()




