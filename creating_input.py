import tensorflow as tf
import scipy.ndimage as sp
import numpy as np
import argparse
import os.path
import global_declare.py
import subprocess


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def get_images():
    images = np.zeros((len(names), IN_HEIGHT, IN_WIDTH, CHANNEL), dtype=np.uint8)
    with open(IMAGE_PATH, 'r') as rf:
        names = rf.readlines()

    for index,value in enumerate(names):
        print('Processing {}'.format(value))
        subprocess.call(['convert', value, '-resize', '448x448', value])
        images[index] = scipy.ndimage.imread(value, mode='RGB')
    
    return images
        

def get_labels():
    labels = np.zeros((len(names), GRID_SIZE, GRID_SIZE, 5+NO_CLASSES), dtype=np.uint8)
    
    with open(LABEL_PATH, 'r') as rf:
        names = rf.readlines()
    
    for index,value in enumerate(names):
        to_store = np.zeros((GRID_SIZE, GRID_SIZE, 5+NO_CLASSES))
        with open(value, 'r') as lf:
            annot = lf.readlines()
            for i in annot:
                words = i.split(' ')
                box = [words[1], words[2], words[3], words[4]]
                x_ind = int(words[1] * GRID_SIZE/IMAGE_SIZE)
                y_ind = int(words[2] * GRID_SIZE/IMAGE_SIZE)
                to_store[x_ind, y_ind, 0] = 1
                to_store[x_ind, y_ind, 1:5] = box
                to_store[x_ind, y_ind, 5 + words[0]] = 1
                
            labels[index] = to_store

    return labels

def convert(imgs, labels, output):
    print('Writing output to {}'.format(output))
    with tf.python_io.TFRecordWriter(output) as writer:
        for label, image in zip(labels, imgs):
            raw = image.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': _bytes_feature(label),
                'image_raw': _bytes_feature(raw)
                }))
            writer.write(example.SerializeToString())
    
            
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', default='result.tfrecords', help='output filename, default to result.tfrecords')

    args = parser.parse.args()
    images = get_images()
    labels = get_labels()
    convert(images, labels, args.output)


if __name__ == '__main__':
    main()
