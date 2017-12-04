import tensorflow as tf
from global_declare import *
import numpy as np

# reading and decoding tf records

def read_and_decode(filename_queue, alter=True):
    with tf.name_scope('read_and_decode'):
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example, features={
            'label': tf.FixedLenFeature([], tf.string),
            'image_raw': tf.FixedLenFeature([], tf.string)
        })

        label = tf.decode_raw(features['label'], tf.uint8)

        image = tf.decode_raw(features['image_raw'], tf.uint8)

        # Convert back to image shape
        image = tf.reshape(image, [IN_HEIGHT, IN_WIDTH, CHANNEL])
        tf.summary.image('real', tf.reshape(image, [1, IN_HEIGHT, IN_WIDTH, CHANNEL]))
        
        label = tf.reshape(label, [GRID_SIZE, GRID_SIZE, 5+NO_CLASSES])
        
        # Add variation
        if alter:
            image = alter_image(image)

        tf.summary.image('altered', tf.reshape(image, [1, IN_HEIGHT, IN_WIDTH, CHANNEL]))

        image = tf.cast(image, tf.float32)
        label = tf.cast(label, tf.float32)
        image = tf.image.per_image_standardization(image)

        return image, label

def alter_image(image):
    with tf.name_scope('alter_image'):
        # Make it slightly bigger and then randomly crop
        altered_image = tf.image.resize_images(image, [IN_HEIGHT+10, IN_WIDTH+10])
        altered_image = tf.random_crop(altered_image, [IN_HEIGHT, IN_WIDTH, CHANNEL])

        # Randomly flip image
        altered_image = tf.image.random_flip_left_right(altered_image)

        # Randomly adjust brightness
        altered_image = tf.image.random_brightness(altered_image, 50)

        return altered_image


