import tensorflow as tf
import scipy.ndimage 
import numpy as np
import argparse
import os.path
from global_declare import *
import subprocess
import os
import cv2


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def get_file_names():
    result = []
    for r,d,f in os.walk('/scratch/ramrao/vehicles/JPEGImages/'):
        for i in f:
            result.append(os.path.join(r,i))
    return result

def get_images(names):
    images = np.zeros((len(names), 448, 448, CHANNEL), dtype=np.uint8)
    
    # Images directory
    os.chdir('/scratch/ramrao/vehicles/JPEGImages/')
    for index,value in enumerate(names):
        value = value[:-1] + '_co.png'
        print('Processing {}'.format(value))
        #subprocess.call(["convert", value, "-resize", "448x448!", value])
        #print(value)
        image = cv2.imread(value)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        images[index] = (image / 255.0)
        #images[index] = scipy.ndimage.imread(value, mode='RGB')
        
    return images
        

def get_labels(names):
    labels = np.zeros((len(names), GRID_SIZE, GRID_SIZE, 5+NO_CLASSES), dtype=np.uint8)
    # Labels directory
    os.chdir('/scratch/ramrao/vehicles/labels/')
    for index,value in enumerate(names):
        to_store = np.zeros((GRID_SIZE, GRID_SIZE, 5+NO_CLASSES))
        #file_val = os.path.basename(os.path.normpath(value))
        file_val = value[:-1] + '_co.txt'
        value = os.path.join('/scratch/ramrao/vehicles/labels',file_val)
        print(value)
        with open(value, 'r') as lf:
            print(value)
            annot = lf.readlines()
            for i in annot:
                words = i.split(' ')
                box = [float(words[1]), float(words[2]), float(words[3]), float(words[4])]
                x_ind = int(float(words[1]) * GRID_SIZE / IMAGE_SIZE)
                y_ind = int(float(words[2]) * GRID_SIZE / IMAGE_SIZE)
                print(x_ind)
                print(y_ind)
                to_store[x_ind, y_ind, 0] = 1
                to_store[x_ind, y_ind, 1:5] = box
                to_store[x_ind, y_ind, 5 + int(words[0])] = 1
                
            labels[index] = to_store
            
    print('Done with labels')
    os.chdir('/scratch/ramrao/vehicles/')
    return labels

def convert(imgs, labels, output):
    print('Writing output to {}'.format(output))
    with tf.python_io.TFRecordWriter(output) as writer:
        for label, image in zip(labels, imgs):
            raw = image.tostring()
            lab_raw = label.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': _bytes_feature(lab_raw),
                'image_raw': _bytes_feature(raw)
                }))
            writer.write(example.SerializeToString())
    
            
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output_file', help='output filename')
    parser.add_argument('input_file', help='input filename')

    args = parser.parse_args()
    #res = get_file_names()
    in_path = os.path.join('/scratch/ramrao/vehicles/folds', args.input_file)
    with open(in_path, 'r') as rf:
        res = rf.readlines()
    images = get_images(res)
    labels = get_labels(res)
    convert(images, labels, args.output_file)


if __name__ == '__main__':
    main()
