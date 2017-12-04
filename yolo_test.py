import tensorflow as tf
import argparse 
from util import *
from global_declare import *
from yolo_model import *
import time
import scipy.ndimage
import os
import cv2
import numpy as np

def detect(res, img, height, width):
    values = interpret_output(res)
    #print(values)
    
    for i in range(len(values)):
        values[i][1] *= (1 * width / IMAGE_SIZE) 
        values[i][1] *= (1 * height / IMAGE_SIZE) 
        values[i][1] *= (1 * width / IMAGE_SIZE) 
        values[i][1] *= (1 * height / IMAGE_SIZE) 

    print(values)
    for i in range(len(values)):
        x = int(values[i][1])
        y = int(values[i][2])
        w = int(values[i][3] / 2)
        h = int(values[i][4] / 2)
        cv2.rectangle(img, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(img, (x - w, y - h - 20),
                      (x + w, y - h), (125, 125, 125), -1)
        cv2.putText(img, values[i][0] + ' : %.2f' % values[i][5], (x - w + 5, y - h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        
    cv2.imwrite('/scratch/ramrao/vehicles/test/image.png', img)
    
    return 1


if __name__ == '__main__':
    # Read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('source', help='list of images for testing')
    parser.add_argument('model', help='model to used')
    args = parser.parse_args()
    
    with open(args.source, 'r') as rf:
        names = rf.readlines()

    X = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, CHANNEL], name='X')
    #y = tf.placeholder(tf.float32, [None, GRID_SIZE, GRID_SIZE, 5+NO_CLASSES], name='labels')

    y_hat = model(X)

    saver = tf.train.Saver()
        
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph('/scratch/ramrao/vehicles/'+args.model + '.meta')
        new_saver.restore(sess, '/scratch/ramrao/vehicles/'+ args.model)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        for index,img in enumerate(names):
            img = os.path.join('/scratch/ramrao/vehicles/JPEGImages/', img[:-1] +'_co.png')
            input_img = cv2.imread(img)
            img_h, img_w, _ = input_img.shape
            inputs = cv2.resize(input_img, (IMAGE_SIZE, IMAGE_SIZE))
            inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB).astype(np.float32)
            inputs = (inputs / 255.0) #* 2.0 - 1.0
            inputs = np.reshape(inputs, (1, IMAGE_SIZE, IMAGE_SIZE, 3))
            
            model_output = sess.run(y_hat, feed_dict={X : inputs})
            #print(model_output.shape)
            detect(model_output[0], input_img, img_h, img_w)
            break
            
        coord.request_stop()
        coord.join(threads)


