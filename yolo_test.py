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
    
    y_hat = model(X)
    result = interpret_output(y_hat)
    
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph('/scratch/ramrao/vehicles/'+args.model + '.meta')
        new_saver.restore(sess, '/scratch/ramrao/vehicles/'+ args.model)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        tf.set_random_seed(1)
        images, labels = sess.run(batch)

        coord.request_stop()
        coord.join(threads)


