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
        

def get_lables():
    labels = np.zeros((len(names), GRID_SIZE, GRID_SIZE, 5+NO_CLASSES), dtype=np.uint8)
    
    with open(LABEL_PATH, 'r') as rf:
        names = rf.readlines()
    
    for index,value in enumerate(nameS):
        to_store = np.zeros((GRID_SIZE, GRID_SIZE, 5+NO_CLASSES))
        with open(value, 'r') as lf:
            annot = lf.readlines()
            for i in annot:
                words = i.split(' ')
                x_ind = int(words[0] * GRID_SIZE/)
                if(to_store[])
                labels[index] = 
            
