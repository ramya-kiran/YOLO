import tensorflow as tf
import argparse 
from util import *
from global_declare import *
from yolo_model import *
import time

if __name__ == '__main__':
    # Read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('train', nargs='+', help='tf record filename')
    parser.add_argument('-b', '--batch-size', default=10, type=int, help='batch size')
    parser.add_argument('-e', '--epochs', default=4000, type=int, help='num of epochs')
    parser.add_argument('-o', '--output', default='model', help='model output')
    parser.add_argument('-l', '--log', default='logs', help='log directory')
    args = parser.parse_args()

    # Reading the tf files and obtaining images and labels
    filename_queue = tf.train.string_input_producer(args.train)
    image, label = read_and_decode(filename_queue)
    batch = tf.train.shuffle_batch([image, label], batch_size=args.batch_size, capacity=800, num_threads=2, min_after_dequeue=200)
    
    # placeholders for input and labels
    X = tf.placeholder(tf.float32, [None, IMAGE_SIZE IMAGE_SIZE, CHANNEL], name='X')
    y = tf.placeholder(tf.float32, [None, GRID_SIZE, GRID_SIZE, 5+NO_CLASSES], name='labels')
    
    # passing the parameters to modela and getting output
    y_hat = model(X)

    total_loss = loss_layer(y_hat, y, args.batch_size)
    tf.summary.scalar('total_loss', total_loss)
    
    global_step = tf.get_variable(
            'global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    learning_rate = tf.train.exponentail_decay(INITIAL_LEARNING_RATE, global_step, DECAY_STEPS, DECAY_RATE, STAIRCASE, name='learning_rate')
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(total_loss, global_step = global_step)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        
        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter(args.log, sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        
        for i in range(args.epochs):
            images, labels = sess.run(batch)
            sess.run(optimizer, feed_dict={X: images, y: labels})
            
            # Print training accuracy every 100 epochs
            if (i+1) % 100 == 0:
                print('loss val {}: {:.2f}'.format(i+1, sess.run(total_loss, feed_dict={X: images, y: labels})))
                
            if (i+1) % 1000 == 0:
                params = saver.save(sess, '{}_{}.ckpt'.format(args.output, i+1))
                print('Model saved: {}'.format(params))
                
        coord.request_stop()
        coord.join(threads)
    
    
