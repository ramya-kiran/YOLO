import tensorflow as tf
import argparse 
from util import *
from global_declare import *
from yolo_model import *
import time


if __name__ == '__main__':
    # Read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('test', help='the testing dataset')
    parser.add_argument('model', help='model to used')
    args = parser.parse_args()

    filename_queue = tf.train.string_input_producer([args.test])
    image, label = read_and_decode(filename_queue)
    batch = tf.train.shuffle_batch([image, label], batch_size=args.batch_size, capacity=300, num_threads=2, seed=1, min_after_dequeue=40)

    X = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, CHANNEL], name='X')
    y = tf.placeholder(tf.float32, [None, GRID_SIZE, GRID_SIZE, 5+NO_CLASSES], name='labels')
