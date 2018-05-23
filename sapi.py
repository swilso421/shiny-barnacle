import numpy as np
import tensorflow as tf

GBL_STDDEV = 0.03

#Creates a convolutional layer for a CNN
def conv_layer(input_data, input_channels, num_filters, filter_shape, name):

    #Organizes the shape of the convolutional layer; used to shape weights
    conv_filter = (filter_shape[0], filter_shape[1], input_channels, num_filters)

    #Generates the weights for each filter
    weights = tf.Variable(tf.truncated_normal(conv_filter, stddev = GBL_STDDEV), name = name + '_W')

    #Generate a bias vector
    bias = tf.Variable(tf.truncated_normal(num_filters), name = name + '_B')

    #Applies the convolution to the input, returning the convolved output
    output = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding = 'SAME')

    #Adds the bias
    output += bias

    #Passes the tensor through a RELU activation layer
    output = tf.nn.relu(output)

    return output

#Creates a pooling layer for downsizing
def pool_layer(input_data, shape = [2, 2]):

    #Shape vector to be used by ksize and strides
    window = [1, shape[0], shape[1], 1]

    #Downsizes the input tensor
    output = tf.nn.max_pool(input_data, ksize = window, strides = window, padding = 'SAME')

    return output
