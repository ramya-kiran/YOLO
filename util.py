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


# convolutional layer to be used by the model 
def convolution_layer(input_imgs, outputs, in_size, stride, name):
    with tf.name_scope('conv_' + name):
        weights = tf.get_variable("w_"+name, [3, 3, in_size, outputs], initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable("b_"+name, [outputs], initializer=tf.constant_initializer(0.0))
        tf.summary.histogram('conv_w_'+name, weights)
        tf.summary.histogram('conv_b_'+name, biases)
        conv = tf.nn.conv2d(input_imgs, weights, strides=[1,stride, stride, 1], padding='SAME', name='conv_w_'+name)
        conv_biased = tf.add(conv, biases, name='conv_b'+name)
        
        return tf.maximum(ALPHA*conv_biased, conv_biased, name='leaky_relu_'+name)


# pooling layer
def pooling_layer(input_tensor, pool_size, pool_stride, name):
    with tf.name_scope('pool_'+name):
        return tf.nn.max_pool(input_tensor, ksize=[1,pool_size, pool_size, 1], strides=[1,pool_stride, pool_stride,1], padding='SAME', name = 'pool_'+name)
        

# fully connected layer
def fc_layer(input_tensor, no_in, no_out, leaky, name):
    with tf.name_scope('fc_'+name):
        weights = tf.get_variable("fc_w_"+name, [no_in, no_out], initializer=tf.contrib.layers.xavier_initializer())
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
    with tf.name_scope('loss_layer'):
        val_1 = GRID_SIZE * GRID_SIZE * NO_CLASSES
        val_2 = val_1 + (GRID_SIZE * GRID_SIZE * NO_BOUNDING_BOX)
        
        pred_classes = tf.reshape(pred[:, : val_1], [batch_size, GRID_SIZE, GRID_SIZE, NO_CLASSES])
        pred_scales = tf.reshape(pred[:, val_1: val_2], [batch_size, GRID_SIZE, GRID_SIZE, NO_BOUNDING_BOX])
        pred_boxes = tf.reshape(pred[:, val_2:], [batch_size, GRID_SIZE, GRID_SIZE, NO_BOUNDING_BOX, 4])
        
        response = tf.reshape(actual[:,:,:,0], [batch_size, GRID_SIZE, GRID_SIZE, 1])
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

        iou_predict_truth = cal_iou(predict_boxes_tran,boxes)

        # calculate I tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        object_mask = tf.reduce_max(iou_predict_truth, 3, keep_dims=True)
        object_mask = tf.cast((iou_predict_truth >= object_mask), tf.float32) * response

        # calculate no_I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        noobject_mask = tf.ones_like(object_mask, dtype=tf.float32) - object_mask
        
        boxes_tran = tf.stack([boxes[:, :, :, :, 0] * GRID_SIZE - offset,
                               boxes[:, :, :, :, 1] * GRID_SIZE - tf.transpose(offset, (0, 2, 1, 3)),
                               tf.sqrt(boxes[:, :, :, :, 2]),
                               tf.sqrt(boxes[:, :, :, :, 3])])
        boxes_tran = tf.transpose(boxes_tran, [1, 2, 3, 4, 0])
        
        # class_loss
        class_delta = response * (pred_classes - classes)
        class_loss = tf.reduce_mean(tf.reduce_sum(tf.square(class_delta), axis=[1, 2, 3]), name='class_loss') * CLASS_SCALE

        # object_loss
        object_delta = object_mask * (pred_scales - iou_predict_truth)
        object_loss = tf.reduce_mean(tf.reduce_sum(tf.square(object_delta), axis=[1, 2, 3]), name='object_loss') * OBJ_SCALE
        
        # noobject_loss
        noobject_delta = noobject_mask * pred_scales
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

        
        
def interpret_output(self, output):
    probs = np.zeros((self.cell_size, self.cell_size,
                      self.boxes_per_cell, self.num_class))
    class_probs = np.reshape(output[0:self.boundary1], (self.cell_size, self.cell_size, self.num_class))
    scales = np.reshape(output[self.boundary1:self.boundary2], (self.cell_size, self.cell_size, self.boxes_per_cell))
    boxes = np.reshape(output[self.boundary2:], (self.cell_size, self.cell_size, self.boxes_per_cell, 4))
    offset = np.transpose(np.reshape(np.array([np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell),
                                         [self.boxes_per_cell, self.cell_size, self.cell_size]), (1, 2, 0))

    boxes[:, :, :, 0] += offset
    boxes[:, :, :, 1] += np.transpose(offset, (1, 0, 2))
    boxes[:, :, :, :2] = 1.0 * boxes[:, :, :, 0:2] / self.cell_size
    boxes[:, :, :, 2:] = np.square(boxes[:, :, :, 2:])
    
    boxes *= self.image_size

    for i in range(self.boxes_per_cell):
        for j in range(self.num_class):
            probs[:, :, i, j] = np.multiply(
                class_probs[:, :, j], scales[:, :, i])

    filter_mat_probs = np.array(probs >= self.threshold, dtype='bool')
    filter_mat_boxes = np.nonzero(filter_mat_probs)
    boxes_filtered = boxes[filter_mat_boxes[0],
                           filter_mat_boxes[1], filter_mat_boxes[2]]
    probs_filtered = probs[filter_mat_probs]
    classes_num_filtered = np.argmax(filter_mat_probs, axis=3)[filter_mat_boxes[
        0], filter_mat_boxes[1], filter_mat_boxes[2]]
    
    argsort = np.array(np.argsort(probs_filtered))[::-1]
    boxes_filtered = boxes_filtered[argsort]
    probs_filtered = probs_filtered[argsort]
    classes_num_filtered = classes_num_filtered[argsort]
    
    for i in range(len(boxes_filtered)):
        if probs_filtered[i] == 0:
            continue
        for j in range(i + 1, len(boxes_filtered)):
            if self.iou(boxes_filtered[i], boxes_filtered[j]) > self.iou_threshold:
                probs_filtered[j] = 0.0
                
    filter_iou = np.array(probs_filtered > 0.0, dtype='bool')
    boxes_filtered = boxes_filtered[filter_iou]
    probs_filtered = probs_filtered[filter_iou]
    classes_num_filtered = classes_num_filtered[filter_iou]

    result = []
    for i in range(len(boxes_filtered)):
        result.append([self.classes[classes_num_filtered[i]], boxes_filtered[i][0], boxes_filtered[
            i][1], boxes_filtered[i][2], boxes_filtered[i][3], probs_filtered[i]])

    return result

def iou(self, box1, box2):
    tb = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - \
         max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
    lr = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - \
         max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
    if tb < 0 or lr < 0:
        intersection = 0
    else:
        intersection = tb * lr
    return intersection / (box1[2] * box1[3] + box2[2] * box2[3] - intersection)
