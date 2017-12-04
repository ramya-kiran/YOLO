import tensorflow as tf
from global_declare import *
import numpy as np
import copy

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


def loss_layer(model_output, ground_truth, batch_size):
    anchors = compute_anchors()
    batch_size, grid_w, grid_h, _ = model_output
    inter_out = np.reshape(model_output, (batch_size, NO_BOUNDING_BOX, 5+NO_CLASSES, GRID_SIZE, GRID_SIZE))
    x,y,w,h,conf,prob = np.split(inter_out, (1,2,3,4,5), axis=2)
    x = tf.sigmoid(x)
    y = tf.sigmoid(y)
    conf = tf.sigmoid(conf)
    prob = np.transpose(prob, (0,2,1,3,4))
    prob = tf.nn.softmax(prob, dim=1)
    
    tw = np.zeros(w.shape, dtype=np.float32)
    th = np.zeros(h.shape, dtype=np.float32)
    tx = np.tile(0.5, x.shape).astype(np.float32)
    ty = np.tile(0.5, y.shape).astype(np.float32)

    box_learning_scale = np.tile(0.1, x.shape).astype(np.float32)
    
    tconf = np.zeros(conf.shape, dtype=np.float32)
    conf_learning_scale = np.tile(0.1, conf.shape).astype(np.float32)
    
    tprob = copy.deepcopy(prob)
    
    x_offset = np.broadcast_to(np.arange(grid_w, dtype=np.float32) x.shape[1:])
    y_offset = np.broadcast_to(np.arange(grid_h, dtype=np.float32).reshape(grid_h, 1), y.shape[1:])

    w_anchor = np.broadcast_to(np.reshape(np.array(anchors, dtype=np.float32)[:, 0], (NO_BOUNDING_BOX, 1, 1, 1)), w.shape[1:])
    h_anchor = np.broadcast_to(np.reshape(np.array(anchors, dtype=np.float32)[:, 1], (NO_BOUNDING_BOX, 1, 1, 1)), h.shape[1:])

    collect_iou = []
    for batch in range(batch_size):
            n_truth_boxes = len(t[batch])
            box_x = (x[batch] + x_offset) / grid_w
            box_y = (y[batch] + y_offset) / grid_h
            box_w = tf.exp(w[batch]) * w_anchor / grid_w
            box_h = tf.exp(h[batch]) * h_anchor / grid_h

            ious = []
            for truth_index in range(n_truth_boxes):
                truth_box_x = Variable(np.broadcast_to(np.array(t[batch][truth_index]["x"], dtype=np.float32), box_x.shape))
                truth_box_y = Variable(np.broadcast_to(np.array(t[batch][truth_index]["y"], dtype=np.float32), box_y.shape))
                truth_box_w = Variable(np.broadcast_to(np.array(t[batch][truth_index]["w"], dtype=np.float32), box_w.shape))
                truth_box_h = Variable(np.broadcast_to(np.array(t[batch][truth_index]["h"], dtype=np.float32), box_h.shape))
                ious.append(multi_box_iou(Box(box_x, box_y, box_w, box_h), Box(truth_box_x, truth_box_y, truth_box_w, truth_box_h)).data.get())  
            ious = np.array(ious)
            best_ious.append(np.max(ious, axis=0))
        best_ious = np.array(best_ious)
    
    
    return total_loss
    
