from keras import backend as K
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.models import Model
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from keras.datasets import mnist
import keras

def cnn_model_fn():

    #input shape = [size, 28, 28, 1]
    input_layer = Input(shape=[28, 28, 1], name='input')

    # Convolutional Layer #1 . out size*32*28*28
    conv1 = Conv2D(filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)(input_layer)
        # tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    # Pooling Layer #1 . out size*32*14*14
    pool1 = MaxPooling2D(pool_size=[2, 2], strides=2)(conv1)
        # tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 . out size*64*14*14
    conv2 = Conv2D(filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)(pool1)
        # tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    # Pooling Layer #2 . out size*64*7*7
    pool2 = MaxPooling2D(pool_size=[2, 2], strides=2)(conv2)
        # tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = Flatten()(pool2)
        # tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = Dense(units=1024, activation=tf.nn.relu)(pool2_flat)
        # tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    # dropout
    dropout = Dropout(rate=0.4)(dense)
        # tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = Dense(units=10, activation='softmax')(dropout)
        # tf.layers.dense(inputs=dropout, units=10)
    model = Model(inputs=input_layer, outputs=logits)
    return model

if __name__=='__main__':
    # prepare data
    # mnist_images = input_data.read_data_sets("MNIST_data/", one_hot=True)

    #shape of x_train is [60000, 28, 28]
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
        x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)
        input_shape = (1, 28, 28)
    else:
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
        input_shape = (28, 28, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    model = cnn_model_fn()
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train,
              batch_size=64,
              epochs=1,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


