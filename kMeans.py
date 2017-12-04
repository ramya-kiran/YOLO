import tensorflow as tf
import numpy as np
import os
import random

NO_CLUSTERS = 5

if __name__ == '__main__':
    main()

def main(input_values):
    with open('/scratch/ramrao/vehicles/kmeans_data.txt') as rf:
        all_data_points = rf.readlines()
    initial_clusters = random.sample(list(range(0, len(all_data_points))), NO_CLUSTERS)
    
    
    return -1

