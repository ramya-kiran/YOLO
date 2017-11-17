from util import *
import tensorflow as tf


def model(input_tensor):
    # [448x448x3] -->  [224x224x16]
    conv_1_out = convolution_layer(input_tensor, 16, 3, 1, '1')
    pool_1_out = pooling_layer(conv_1_out, 2, 2, '1')
    
    # [224x224x16] --> [112x112x32]
    conv_2_out = convolution_layer(pool_1_out, 32, 16, 1, '2')
    pool_2_out = pooling_layer(conv_2_out, 2, 2, '2')
    
    # [112x112x32] --> [56x56x64]
    conv_3_out = convolution_layer(pool_2_out, 64, 32, 1, '3')
    pool_3_out = pooling_layer(conv_3_out, 2, 2, '3')
    
    # [56x56x64] --> [28x28x128]
    conv_4_out = convolution_layer(pool_3_out, 128, 64, 1, '4')
    pool_4_out = pooling_layer(conv_4_out, 2, 2, '4')

    # [28x28x128] --> [14x14x256]
    conv_5_out = convolution_layer(pool_4_out, 256, 128, 1, '5')
    pool_5_out = pooling_layer(conv_5_out, 2, 2, '5')
    
    # [14x14x256] --> [7x7x512]
    conv_6_out = convolution_layer(pool_5_out, 512, 256, 1, '6')
    pool_6_out = pooling_layer(conv_6_out, 2, 2, '6')
    
    # [7x7x512] --> [7x7x1024]
    conv_7_out = convolution_layer(pool_6_out, 1024, 512, 1, '7')
    
    # [7x7x1024] --> [7x7x1024]
    conv_8_out = convolution_layer(conv_7_out, 1024, 1024, 1, '8')
    
    # output = [batch_size, 7*7*1024]
    flatten_1 = tf.reshape(conv_8_out, [-1, 7*7*1024])
    
    # with tf.Session() as sess:
    #     print(sess.run(tf.shape(flatten_1)))
    fc_1_out = fc_layer(flatten_1, 7*7*1024, 256, True, '9')
    fc_2_out = fc_layer(fc_1_out, 256, 256, True, '10')
    fc_3_out = fc_layer(fc_2_out, 4096, 256, True, '11')

    return fc_3_out
    
