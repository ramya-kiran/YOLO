import tensorflow as tf
import scipy.ndimage as sp
import numpy as np
import argparse
import os.path

IN_HEIGHT = 512
IN_WIDTH= 512
IN_CHANNEL = 3

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert(filenames):
    with open(filename, 'r') as rf:
        names = rf.readlines()
        images = np.zeros((len(names), IN_HEIGHT, IN_WIDTH, CHANNEL), dtype=np.uint8)
        labels = np.zeros((len(names), IN_HEIGHT, IN_WIDTH, CHANNEL), dtype=np.uint8)
        for i in names:
            print('Processing filename {}'.format(filename))
            
