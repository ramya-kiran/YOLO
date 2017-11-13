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

def cal_iou(boxes1, boxes2):
    with tf.variable_scope('iou'):
        boxes1 = tf.stack([boxes1[:, :, :, :, 0] - boxes1[:, :, :, :, 2] / 2.0,
                           boxes1[:, :, :, :, 1] - boxes1[:, :, :, :, 3] / 2.0,
                           boxes1[:, :, :, :, 0] + boxes1[:, :, :, :, 2] / 2.0,
                           boxes1[:, :, :, :, 1] + boxes1[:, :, :, :, 3] / 2.0])
        boxes1 = tf.transpose(boxes1, [1, 2, 3, 4, 0])
        
        boxes2 = tf.stack([boxes2[:, :, :, :, 0] - boxes2[:, :, :, :, 2] / 2.0,
                           boxes2[:, :, :, :, 1] - boxes2[:, :, :, :, 3] / 2.0,
                           boxes2[:, :, :, :, 0] + boxes2[:, :, :, :, 2] / 2.0,
                           boxes2[:, :, :, :, 1] + boxes2[:, :, :, :, 3] / 2.0])
        boxes2 = tf.transpose(boxes2, [1, 2, 3, 4, 0])

        # calculate the left up point & right down point
        lu = tf.maximum(boxes1[:, :, :, :, :2], boxes2[:, :, :, :, :2])
        rd = tf.minimum(boxes1[:, :, :, :, 2:], boxes2[:, :, :, :, 2:])
        
        # intersection
        intersection = tf.maximum(0.0, rd - lu)
        inter_square = intersection[:, :, :, :, 0] * intersection[:, :, :, :, 1]
        
        # calculate the boxs1 square and boxs2 square
        square1 = (boxes1[:, :, :, :, 2] - boxes1[:, :, :, :, 0]) * \
                  (boxes1[:, :, :, :, 3] - boxes1[:, :, :, :, 1])
        square2 = (boxes2[:, :, :, :, 2] - boxes2[:, :, :, :, 0]) * \
                  (boxes2[:, :, :, :, 3] - boxes2[:, :, :, :, 1])

        union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)

        return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)



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

        # calculate I tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        object_mask = tf.reduce_max(iou_predict_truth, 3, keep_dims=True)
        object_mask = tf.cast((iou_predict_truth >= object_mask), tf.float32) * response

        # calculate no_I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        noobject_mask = tf.ones_like(object_mask, dtype=tf.float32) - object_mask
        
        boxes_tran = tf.stack([boxes[:, :, :, :, 0] * self.cell_size - offset,
                               boxes[:, :, :, :, 1] * self.cell_size - tf.transpose(offset, (0, 2, 1, 3)),
                               tf.sqrt(boxes[:, :, :, :, 2]),
                               tf.sqrt(boxes[:, :, :, :, 3])])
        boxes_tran = tf.transpose(boxes_tran, [1, 2, 3, 4, 0])
        
        # class_loss
        class_delta = response * (pred_classes - classes)
        class_loss = tf.reduce_mean(tf.reduce_sum(tf.square(class_delta), axis=[1, 2, 3]), name='class_loss') * CLASS_SCALE

        # object_loss
        object_delta = object_mask * (predict_scales - iou_predict_truth)
        object_loss = tf.reduce_mean(tf.reduce_sum(tf.square(object_delta), axis=[1, 2, 3]), name='object_loss') * OBJ_SCALE
        
        # noobject_loss
        noobject_delta = noobject_mask * predict_scales
        noobject_loss = tf.reduce_mean(tf.reduce_sum(tf.square(noobject_delta), axis=[1, 2, 3]), name='noobject_loss') * NOOBJ_SCALE

        # coord_loss
        coord_mask = tf.expand_dims(object_mask, 4)
        boxes_delta = coord_mask * (pred_boxes - boxes_tran)
        coord_loss = tf.reduce_mean(tf.reduce_sum(tf.square(boxes_delta), axis=[1, 2, 3, 4]), name='coord_loss') * COORD_SCALE

        tf.losses.add_loss(class_loss)
        tf.losses.add_loss(object_loss)
        tf.losses.add_loss(noobject_loss)
        tf.losses.add_loss(coord_loss)

        tf.summary.scalar('class_loss', class_loss)
        tf.summary.scalar('object_loss', object_loss)
        tf.summary.scalar('noobject_loss', noobject_loss)
        tf.summary.scalar('coord_loss', coord_loss)
        
        tf.summary.histogram('boxes_delta_x', boxes_delta[:, :, :, :, 0])
        tf.summary.histogram('boxes_delta_y', boxes_delta[:, :, :, :, 1])
        tf.summary.histogram('boxes_delta_w', boxes_delta[:, :, :, :, 2])
        tf.summary.histogram('boxes_delta_h', boxes_delta[:, :, :, :, 3])
        tf.summary.histogram('iou', iou_predict_truth)
        
        return tf.losses.get_total_loss()

        
            
