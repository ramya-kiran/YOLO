from util import *
import tensorflow as tf
import tensorflow.contrib.slim as slim


def model(input_tensor):
    # [448x448x3] -->  [224x224x16]
    conv_1_out = convolution_layer(input_tensor, 16, 3,3, 1, '1')
    conv_1_out = slim.batch_norm(conv_1_out, center=True, scale=True, epsilon=1e-5, is_training=True)
    pool_1_out = pooling_layer(conv_1_out, 2, 2, '1')
    
    # [224x224x16] --> [112x112x32]
    conv_2_out = convolution_layer(pool_1_out, 32, 16, 3,1, '2')
    conv_2_out = slim.batch_norm(conv_2_out, center=True, scale=True, epsilon=1e-5, is_training=True)
    pool_2_out = pooling_layer(conv_2_out, 2, 2, '2')
    
    # [112x112x32] --> [56x56x64]
    conv_3_out = convolution_layer(pool_2_out, 64, 32, 3,1, '3')
    conv_3_out = slim.batch_norm(conv_3_out, center=True, scale=True, epsilon=1e-5, is_training=True)
    pool_3_out = pooling_layer(conv_3_out, 2, 2, '3')

    # the convolutions with filter sizes 3x3 and 1x1 --> 56x56x64
    conv_4_out = convolution_layer(pool_3_out, 128, 64, 3,1, '4')
    conv_4_out = slim.batch_norm(conv_4_out, center=True, scale=True, epsilon=1e-5, is_training=True)
    conv_5_out = convolution_layer(conv_4_out, 64, 128, 1,1, '5')
    conv_5_out = slim.batch_norm(conv_5_out, center=True, scale=True, epsilon=1e-5, is_training=True)
    
    # [56x56x64] --> [28x28x128]
    conv_6_out = convolution_layer(conv_5_out, 128, 64, 3,1, '6')
    conv_6_out = slim.batch_norm(conv_6_out, center=True, scale=True, epsilon=1e-5, is_training=True)
    pool_4_out = pooling_layer(conv_6_out, 2, 2, '4')

    # the convolutions with filter sizes 3x3 and 1x1 --> 28x28x128
    conv_7_out = convolution_layer(pool_4_out, 256, 128, 3,1, '7')
    conv_7_out = slim.batch_norm(conv_7_out, center=True, scale=True, epsilon=1e-5, is_training=True)
    conv_8_out = convolution_layer(conv_7_out, 128, 256, 1,1, '8')
    conv_8_out = slim.batch_norm(conv_8_out, center=True, scale=True, epsilon=1e-5, is_training=True)

    # [28x28x128] --> [14x14x256]
    conv_9_out = convolution_layer(conv_8_out, 256, 128,3, 1, '9')
    conv_9_out = slim.batch_norm(conv_9_out, center=True, scale=True, epsilon=1e-5, is_training=True)
    pool_5_out = pooling_layer(conv_9_out, 2, 2, '5')

    # the convolutions with filter sizes 3x3 and 1x1 --> 14x14x256
    conv_10_out = convolution_layer(pool_5_out, 512, 256, 3,1, '10')
    conv_10_out = slim.batch_norm(conv_10_out, center=True, scale=True, epsilon=1e-5, is_training=True)
    conv_11_out = convolution_layer(conv_10_out, 256, 512, 1,1, '11')
    conv_11_out = slim.batch_norm(conv_11_out, center=True, scale=True, epsilon=1e-5, is_training=True)
    conv_12_out = convolution_layer(conv_11_out, 512, 256, 3,1, '12')
    conv_12_out = slim.batch_norm(conv_12_out, center=True, scale=True, epsilon=1e-5, is_training=True)
    conv_13_out = convolution_layer(conv_12_out, 256, 512, 1,1, '13')
    conv_13_out = slim.batch_norm(conv_13_out, center=True, scale=True, epsilon=1e-5, is_training=True)
    
    # [14x14x256] --> [7x7x512]
    conv_14_out = convolution_layer(conv_13_out, 512, 256, 3,1, '14')
    conv_14_out = slim.batch_norm(conv_14_out, center=True, scale=True, epsilon=1e-5, is_training=True)
    pool_6_out = pooling_layer(conv_14_out, 2, 2, '6')
    
    # the convolutions with filter sizes 3x3 and 1x1 --> 7x7x1024
    conv_15_out = convolution_layer(pool_6_out, 1024, 512, 3,1, '15')
    conv_15_out = slim.batch_norm(conv_15_out, center=True, scale=True, epsilon=1e-5, is_training=True)
    conv_16_out = convolution_layer(conv_15_out, 512, 1024, 1,1, '16')
    conv_16_out = slim.batch_norm(conv_16_out, center=True, scale=True, epsilon=1e-5, is_training=True)
    conv_17_out = convolution_layer(conv_16_out, 1024, 512, 3,1, '17')
    conv_17_out = slim.batch_norm(conv_17_out, center=True, scale=True, epsilon=1e-5, is_training=True)
    conv_18_out = convolution_layer(conv_17_out, 512, 1024, 1,1, '18')
    conv_18_out = slim.batch_norm(conv_18_out, center=True, scale=True, epsilon=1e-5, is_training=True)
    conv_19_out = convolution_layer(conv_18_out, 1024, 512, 3,1, '19')
    conv_19_out = slim.batch_norm(conv_19_out, center=True, scale=True, epsilon=1e-5, is_training=True)

    # Detection network
    conv_20_out = convolution_layer(conv_19_out, 1024, 1024, 3,1, '20')
    conv_20_out = slim.batch_norm(conv_20_out, center=True, scale=True, epsilon=1e-5, is_training=True)
    conv_21_out = convolution_layer(conv_20_out, TOTAL_OUTPUTS, 1024, 1,1, '21')
    conv_21_out = slim.batch_norm(conv_21_out, center=True, scale=True, epsilon=1e-5, is_training=True)
    # conv_22_out = convolution_layer(conv_21_out, 1024, TOTAL_OUTPUTS, 3,1, '22')
    # conv_23_out = convolution_layer(conv_22_out, TOTAL_OUTPUTS, 1024, 1,1, '23')
    # conv_24_out = convolution_layer(conv_23_out, 1024, TOTAL_OUTPUTS, 3,1, '24')
    # conv_24_out = convolution_layer(conv_24_out, TOTAL_OUTPUTS, 1024, 1,1, '25')
    
    final_output = tf.reshape(conv_21_out, [-1, 7*7*TOTAL_OUTPUTS])

    fc_1_out = fc_layer(final_output, 7*7*TOTAL_OUTPUTS, TOTAL_OUTPUTS, True, '1')

    return fc_1_out
    

# convolutional layer to be used by the model                                                                         
def convolution_layer(input_imgs, outputs, in_fil, in_size, stride, name):                                                    
    with tf.name_scope('conv_' + name):                                                                               
        weights = tf.get_variable("w_"+name, [in_size, in_size, in_fil, outputs], initializer=tf.contrib.layers.xavier_initializer())        
        biases = tf.get_variable("b_"+name, [outputs], initializer=tf.constant_initializer(0.0))
        tf.summary.histogram('conv_w_'+name, weights)                                                                 
        tf.summary.histogram('conv_b_'+name, biases)                                                                  
        conv = tf.nn.conv2d(input_imgs, weights, strides=[1,stride, stride, 1], padding='SAME', name='conv_w_'+name)  
        conv_biased = tf.add(conv, biases, name='conv_b'+name)                                                        
                                                                                                                      
        return tf.maximum(ALPHA*conv_biased, conv_biased, name='leaky_relu_'+name)                                    
                                                                                                                      
                                                                                                                      
# pooling layer                                                                                                       
def pooling_layer(input_tensor, pool_size, pool_stride, name):                                                        
    with tf.name_scope('pool_'+name):                                                                                 
        return tf.nn.max_pool(input_tensor, ksize=[1,pool_size, pool_size, 1], strides=[1,pool_stride, pool_stride,1]\
, padding='SAME', name = 'pool_'+name)                                                                                
                                                                                                                      
                                                                                                                      
# fully connected layer                                                                                               
def fc_layer(input_tensor, no_in, no_out, leaky, name):                                                               
    with tf.name_scope('fc_'+name):                                                                                   
        weights = tf.get_variable("fc_w_"+name, [no_in, no_out], initializer=tf.contrib.layers.xavier_initializer())  
        biases = tf.get_variable("fc_b_"+name, [no_out], initializer=tf.constant_initializer(0.0))                    
        y = tf.add(tf.matmul(input_tensor, weights), biases)                                                          
        tf.summary.histogram('fc_w'+name, weights)                                                                    
        tf.summary.histogram('fc_b'+name, biases)                                                                     
                                                                                                                      
        return(tf.maximum(ALPHA*y, y, name='fc_out'+name))                
