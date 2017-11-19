# YOLO

This repositiory contains the implementation of YOLO version V1 using python version 3.6 and tensorflow version 1.3. 

creating_input.py : Creates tf records with images and its corresponding labels. Using the VEDAI dataset and converting 
annotations to the form as shown below

img1 --> annotation 1 -> x_center, y_center, width, height 

global_declare.py: This file contains all the global variables used for trainign the network. 

util.py:  Code for building the model and calculating the loss is present in this file. 

yolo_model.py : The model is built in this file.

yolo_train.py : This file contains code to train the network.

To run: python yolo_train path_to_.tfrecords -b batch_size -e epochs -o where_to_store_trained weigths -l to_store_evennt_files

