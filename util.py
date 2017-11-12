import tensorflow as tf
import global_declare.py
ALPHA = 0.1


def convolution_layer(input_imgs, outputs, size, stride, name):
    with tf.name_scope('conv_' + name):
        channels = input_imgs.get_shape()[3]
        weights = tf.get_variable("w_"+name, [size, size, channels, outputs], initializer=tf.contrib.layers.xavier_initialization())
        biases = tf.get_variable("b_"+name, [outputs], initializer=tf.constant_initializer(0.0))
        tf.summary.histogram('conv_w_'+name, weights)
        tf.summary.histogram('conv_b_'+name, biases)
        conv = tf.nn.conv2d(input_imgs, weights, strides[1,stride, stride, 1], padding='SAME', name='conv_w_'+name)
        conv_biased = tf.add(conv, biases, name='conv_b'+name)
        
        return tf.maximum(ALPHA*conv_biased, conv_biased, name='leaky_relu_'+name)


def pooling_layer(input_tensor, pool_size, pool_stride, name):
    with tf.name_scope('pool_'+name):
        return tf.nn.max_pool(input_tensor, ksize=[1,pool_size, pool_size, 1], strides=[1,pool_stride, pool_stride,1], padding='SAME', name = 'pool_'+name)


def fc_layer(input_tensor, no_out, no_in, leaky):
    with tf.name_scope('fc_'+name):
        weights = tf.get_variable("fc_w_"+name, [no_in, no_out], initializer=tf.contrib.layers.xavier_initialization())
        biases = tf.get_variable("fc_b_"+name, [no_out], initializer=tf.constant_initializer(0.0))
        y = tf.add(tf.matmul(input_tensor, weights), biases)
        tf.summary.histogram('fc_w'+name, weights)
        tf.summary.histogram('fc_b'+name, biases)
        
        return(tf.maximum(ALPHA*y, y, name='fc_out'+name))
