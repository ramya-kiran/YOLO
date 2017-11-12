import tensorflow as tf
from global_declare import *


# convolutional layer to be used by the model 
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


# pooling layer
def pooling_layer(input_tensor, pool_size, pool_stride, name):
    with tf.name_scope('pool_'+name):
        return tf.nn.max_pool(input_tensor, ksize=[1,pool_size, pool_size, 1], strides=[1,pool_stride, pool_stride,1], padding='SAME', name = 'pool_'+name)


# fully connected layer
def fc_layer(input_tensor, no_out, no_in, leaky):
    with tf.name_scope('fc_'+name):
        weights = tf.get_variable("fc_w_"+name, [no_in, no_out], initializer=tf.contrib.layers.xavier_initialization())
        biases = tf.get_variable("fc_b_"+name, [no_out], initializer=tf.constant_initializer(0.0))
        y = tf.add(tf.matmul(input_tensor, weights), biases)
        tf.summary.histogram('fc_w'+name, weights)
        tf.summary.histogram('fc_b'+name, biases)
        
        return(tf.maximum(ALPHA*y, y, name='fc_out'+name))


# loss layer
def loss_layer(pred, actual, batch_size):
    with tf.name_Scope('loss_layer'):
        val_1 = GRID_SIZE * GRID_SIZE * NO_CLASSES
        val_2 = val_1 + (GRID_SIZE * GRID_SIZE * NO_BOUNDING_BOX)
        pred_classes = np.reshape(pred[:, : val_1], [batch_size, GRID_SIZE, GRID_SIZE, NO_CLASSES])
        pred_scales = np.reshape(pred[:, val_1: val_2], [batch_size, GRID_SIZE, GRID_SIZE, NO_BOUNDING_BOX])
        pred_boxes = np.reshape(pred[:, val_2:], [batch_size, GRID_SIZE, GRID_SIZE, NO_BOUNDING_BOX, 4])
        
        response = tf.reshape(actual[:,:,:,0], [batch_size, GRID_SIZE, GRID_SIZE, NO_BOUNDING_BOX, 1])
        boxes = tf.reshape(actual[:,:,:,1:5], [batch_size,GRID_SIZE, GRID_SIZE, 1, 4])
        boxes = tf.tile(boxes, [1,1,1, NO_BOUNDING_BOX, 1]) / IMAGE_SIZE
        classes = actual[:,:,:,5:]

        offset = np.transpose(np.reshape(np.array(
            [np.arange(GRID_SIZE)] * GRID_SIZE * NO_BOUNDING_BOX),
            (NO_BOUNDING_BOX, GRID_SIZE, GRID_SIZE)), (1, 2, 0))
        
        offset = tf.constant(offset, dtype=tf.float32)
        offset = tf.reshape(offset, [1, GRID_SIZE, GRID_SIZE, NO_BOUNDING_BOX])
        offset = tf.tile(offset, [batch_size, 1, 1, 1])
        predict_boxes_tran = tf.stack([(pred_boxes[:, :, :, :, 0] + offset) / GRID_SIZE,
                (pred_boxes[:, :, :, :, 1] + tf.transpose(offset, (0, 2, 1, 3))) / GRID_SIZE,
                tf.square(pred_boxes[:, :, :, :, 2]),
                tf.square(pred_boxes[:, :, :, :, 3])])
        predict_boxes_tran = tf.transpose(predict_boxes_tran, [1, 2, 3, 4, 0])

        iou_predict_truth = cal_iou(predict_box_tran,boxes)

        
