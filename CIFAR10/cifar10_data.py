# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 22:13:00 2018

@author: abra
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile
import glob

from six.moves import urllib
import tensorflow as tf

DATA_DIR='.\CIFAR10-data'
DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
LABEL_SIZE = 1
DATA_HEIGHT = 32
DATA_WIDTH = 32
DATA_DEPTH = 3

def maybe_download_and_extract():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(DATA_DIR, filename)
    print('filepath: ' + filepath)
    if not os.path.exists(filepath):
        filepath = urllib.request.urlretrieve(DATA_URL, filepath)

    extracted_dir_path = os.path.join(DATA_DIR, 'cifar-10-batches-bin')
    if not os.path.exists(extracted_dir_path):
        t = tarfile.open(filepath, 'r:gz')
        t.extractall(DATA_DIR)
        
def __input(filenames):
    IMAGE_SIZE = DATA_HEIGHT * DATA_WIDTH * DATA_DEPTH
    RECORD_SIZE = LABEL_SIZE + IMAGE_SIZE
    
    dataset = tf.contrib.data.FixedLengthRecordDataset(filenames, record_bytes=RECORD_SIZE)
    def _parse_function(value):
        record_bytes = tf.decode_raw(value, tf.uint8)
        label = tf.cast(tf.strided_slice(record_bytes, [0], [LABEL_SIZE]), tf.int32)
        image = tf.reshape(tf.strided_slice(record_bytes, [LABEL_SIZE], [RECORD_SIZE]), [DATA_DEPTH, DATA_HEIGHT, DATA_WIDTH])
        # Convert from [depth, height, width] to [height, width, depth].
        image = tf.transpose(image, [1,2,0])
        image = tf.image.per_image_standardization(image)
        return label, image
    dataset = dataset.map(_parse_function)
    return dataset        

def train_input():
    data_dir = os.path.join(DATA_DIR, 'cifar-10-batches-bin')    
    filenames = []
    for fname in glob.glob(data_dir + '\data_batch*'):
        filenames.append(fname)
    return __input(filenames)
        
def test_input():
    data_dir = os.path.join(DATA_DIR, 'cifar-10-batches-bin')
    filenames = []
    for fname in glob.glob(data_dir + '\test_batch*'):
        filenames.append(fname)
    return __input(filenames)