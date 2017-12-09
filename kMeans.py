import tensorflow as tf
import numpy as np
import os
import random

NO_CLUSTERS = 5


def IOU(data_value, clusters):
    data_half = data_value/2
    half_centroids = clusters/2
    left_w = np.maximum(-half_centroids[:,0], -np.tile(data_half, (NO_CLUSTERS,1))[:,0])
    right_w = np.minimum(half_centroids[:,0], np.tile(data_half, (NO_CLUSTERS,1))[:,0])
    w_overlap = right_w - left_w
    
    left_h = np.maximum(-half_centroids[:,1], -np.tile(data_half, (NO_CLUSTERS,1))[:,1])
    right_h = np.minimum(half_centroids[:,1], np.tile(data_half, (NO_CLUSTERS,1))[:,1])
    h_overlap = right_h - left_h
    
    intersection = w_overlap * h_overlap
    
    new_data = np.tile(data_value, (NO_CLUSTERS,1))
    union = (clusters[:,0] * clusters[:,1]) + (new_data[:,0] * new_data[:,1]) - intersection
    
    #print(intersection/union)
    final_value = 1 - (intersection / union)
    
    return np.argmin(final_value) 
    

def kmeans_llyod(data_points, cluster_indexes):
    N, p = data_points.shape
    #cluster_centroids = data_points[cluster_indexes, :]
    s = data_points[cluster_indexes, :]
    print(s)
    data_points = np.concatenate((data_points, np.zeros((N, 1))), axis=1)
    data_points, v= kmeans_iterations(data_points, s)
    cluster_centroids = v
    prev_clusters = np.array(data_points[:,2])
    print(prev_clusters)
    no_iter = 0
    check = 0
    while check != 1:
        data_points, cluster_centroids = kmeans_iterations(data_points, cluster_centroids)
        next_clusters = np.array(data_points[:,2])
        print(next_clusters)
        no_iter += 1
        print(no_iter)
        print(sum(next_clusters == prev_clusters))
        if(sum(next_clusters == prev_clusters) != N):
            prev_clusters = next_clusters
            print("Inside not equal to")
            print(sum(next_clusters == prev_clusters) != N)
        else:
            check = 1
    return cluster_centroids


def kmeans_iterations(data_points, cluster_means):
    N, p = data_points.shape
    print(N)
    for i in range(N):
        data_points[i,2] = IOU(data_points[i,:2], cluster_means)

    for m in range(NO_CLUSTERS):
        w_collect = sum(data_points[data_points[:,2] == m, 0])
        print(str(sum(data_points[:,2] == m)) + ' ' + str(m))
        w_collect = w_collect / sum(data_points[:,2] == m)
        h_collect = sum(data_points[data_points[:,2] == m, 1])
        h_collect = h_collect / sum(data_points[:,2] == m)
        cluster_means[m,0] = w_collect
        cluster_means[m,1] = h_collect
        
    return data_points, cluster_means
    
def main():
    with open('/scratch/ramrao/vehicles/kmeans_data.txt') as rf:
        all_data_points = rf.readlines()
    
    collect_points = []
    for d in all_data_points:
        collect_points.append(list(map(float, d.split(' '))))

    collect_points = np.array(collect_points)
    initial_clusters = []
    initial_indexes = random.sample(list(range(0, len(all_data_points))), NO_CLUSTERS)
    print(initial_indexes)
    
    res_clusters = kmeans_llyod(collect_points, initial_indexes)
        
    print("Result")
    print(res_clusters)
    return 1

if __name__ == '__main__':
    main()
