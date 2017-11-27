import tensorflow as tf
import argparse 
from util import *
from global_declare import *
from yolo_model import *
import time
import numpy as np

def detect(res, img, heights, width):
    values = interpret_output(res)
    
    for i in range(len(values)):
        values[i][1] *= (1 * width / IMAGE_SIZE) 
        values[i][1] *= (1 * height / IMAGE_SIZE) 
        values[i][1] *= (1 * width / IMAGE_SIZE) 
        values[i][1] *= (1 * height / IMAGE_SIZE) 

    for i in range(len(values)):
        x = int(values[i][1])
        y = int(values[i][2])
        w = int(values[i][3] / 2)
        h = int(values[i][4] / 2)
        cv2.rectangle(img, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(img, (x - w, y - h - 20),
                      (x + w, y - h), (125, 125, 125), -1)
        cv2.putText(img, result[i][0] + ' : %.2f' % result[i][5], (x - w + 5, y - h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.CV_AA)

    imshow('Image', img)
    cv2.waitKey(wait)
    imwrite('/scratch/ramrao/test/', img)
    
    return 1


if __name__ == '__main__':
    # Read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('source', help='list of images for testing')
    parser.add_argument('model', help='model to used')
    args = parser.parse_args()
    
    with open(args.source, 'r') as rf:
        names = readlines()

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
            img = os.path.join(args.source, img +'_co.png')
            input_img = scipy.ndimage.imread(img, mode='RGB')
            img_h, img_w, _ = img.shape
            inputs = cv2.resize(img, (self.image_size, self.image_size))
            inputs = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB).astype(np.float32)
            inputs = (inputs / 255.0) * 2.0 - 1.0
            inputs = np.reshape(inputs, (1, self.image_size, self.image_size, 3))
            
            model_ouptut = sess.run(y_hat, feed_dict={X : inputs})
            detect(model_output, img, img_h, img_w)
            
            sess.run(result, feed_dict={X:input_img})
            detect(result, input_img)
            
        coord.request_stop()
        coord.join(threads)


