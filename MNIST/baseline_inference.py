import tensorflow as tf
import logging
import os

logging.basicConfig(level = logging.DEBUG)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("ckpt", '/home/nineg/PycharmProjects/nerdnerdML/MNIST/2018_Mar_31_13_37_29/200000-step_0.01-learningrate_64-batchsize/parameters-185000', "previous model and parameter")
tf.app.flags.DEFINE_string("image", '/home/nineg/PycharmProjects/nerdnerdML/MNIST/MNIST_predict_image', "image path")
tf.app.flags.DEFINE_string("answer", '/home/nineg/PycharmProjects/nerdnerdML/MNIST/MNIST_predict_image.txt', "image answer")


def _load_graph(ckpt):
    meta_file = ckpt + ".meta"
    tf.train.import_meta_graph(meta_file)


def _load_parameter(sess, ckpt):
    saver = tf.train.Saver()
    if os.path.isdir(ckpt):
        ckpt = tf.train.latest_checkpoint(ckpt)
    saver.restore(sess, ckpt)


def get_image():
    """
    get name and image data from FLAGS.image path

    :return name: filename
            data: image data with shape [None, 784]

    """
    def _single_image(filename):
        # Read the filename:
        # image_data = tf.gfile.FastGFile(filenames[i], 'rb').read()
        image_file = tf.gfile.FastGFile(filename, 'rb').read()
        image = tf.image.decode_jpeg(image_file)
        # image_data = tf.image.resize_images(image, [28, 28])
        # image_data = tf.image.convert_image_dtype(image_data, dtype=tf.uint8)
        image_data = tf.reshape(image, [784])
        # image_data = tf.cast(image_data, dtype=tf.uint8)
        image_data = tf.cast(image_data, dtype=tf.float32)
        image_data = tf.divide(image_data, 127.5)
        return image_data.eval()

    name = []
    data = []
    if os.path.isdir(FLAGS.image):
        with tf.Session():
            for file_name in os.listdir(FLAGS.image):
                file_path = os.path.join(FLAGS.image, file_name)
                data.append(_single_image(file_path))
                name.append(file_name)

    return name, data


def get_image_anser(answer):
    """
    get labels of image
    :param answer: the file name
    :return: dict: {fliename:label}
    """
    dict = {}
    with open(answer, mode='r') as f:
        for line in f.read().splitlines():
            dict.update({line.split(',')[0].strip(): line.split(',')[1].strip()})
    return dict


def main(unused):
    _load_graph(FLAGS.ckpt)

    graph = tf.get_default_graph()
    ops = graph.get_operations()
    for op in ops:
        logging.debug("op name: %s", op.name)

    """
    load input and logit tensor from graph
    """
    input = graph.get_tensor_by_name('input_data:0')
    logit = graph.get_tensor_by_name('logits/add:0')

    prob = tf.nn.softmax(logit)
    predict = tf.argmax(prob, axis=1)

    name_list, data_list = get_image()

    with tf.Session() as session:
        _load_parameter(session, FLAGS.ckpt)

        result = session.run(predict, feed_dict={input: data_list})

    logging.debug("result %s", result)

    """
    calculate result hit rate
    """
    list = zip(name_list, result)
    # for z in list:
    #     logging.debug("list %s", z[0])

    answers = get_image_anser(FLAGS.answer)
    # for answer in answers:
    #     logging.debug("answer %s,%s", answer, answers[answer])

    correct_count = 0
    for z in list:
        logging.debug("score %s, %s, %s, %s", z[0], z[1], answers[z[0]], str(z[1]) == answers[z[0]])
        correct_count += 1 if str(z[1]) == answers[z[0]] else 0

    logging.debug("correct %s, rate %s", correct_count, float(correct_count) / len(result))

if __name__ == '__main__':
    tf.app.run()