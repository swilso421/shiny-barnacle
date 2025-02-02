import numpy as np
import tensorflow as tf

GBL_STDDEV = 0.03 #Stddev for all random weight initializations

#Creates a convolutional layer for a CNN
def conv_layer(input_data, input_channels, num_filters, filter_shape, name = 'CONV'):
    '''Creates a convolutional layer for a CNN'''

    #Organizes the shape of the convolutional layer; used to shape weights
    conv_filter = (filter_shape[0], filter_shape[1], input_channels, num_filters)

    #Generates the weights for each filter
    weights = tf.Variable(tf.truncated_normal(conv_filter, stddev = GBL_STDDEV), name = name + '_W')

    #Generate a bias vector
    bias = tf.Variable(tf.truncated_normal(num_filters), name = name + '_B')

    #Applies the convolution to the input, returning the convolved output
    output = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding = 'SAME', name = name)

    #Adds the bias
    output += bias

    #Passes the tensor through a RELU activation layer
    output = tf.nn.relu(output, name = name + '_RELU')

    return output

#Creates a pooling layer for downsizing
def pool_layer(input_data, shape = [2, 2], name = 'POOL'):
    '''Creates a pooling layer for downsizing'''

    #Shape vector to be used by ksize and strides
    window = [1, shape[0], shape[1], 1]

    #Downsizes the input tensor
    output = tf.nn.max_pool(input_data, ksize = window, strides = window, padding = 'SAME', name = name)

    return output

#Creates a fully connected layer for classification
def fc_layer(input_data, num_nodes, dropout = 0.0, relu = True, name = 'FC'):
    '''Creates a fully connected layer for classification'''

    #Generate the weights for each neuron
    weights = tf.Variable(tf.truncated_normal([len(input_data[1]), num_nodes], stddev = GBL_STDDEV), name = name + '_W')

    #Generate a bias vector
    bias = tf.Variable(tf.truncated_normal(num_nodes), name = name + '_B')

    #Multiply the input by the neuron weights and add the neuron biases
    output = tf.matmul(input_data, weights, name = name) + bias

    #Apply RELU non-linearity
    if relu:
        output = tf.nn.relu(output, name = name + '_RELU')

    #If dropout in valid range, apply dropout
    if 0.0 < droput < 1.0:
        output = tf.nn.dropout(output, keep_prob = dropout, name = name + '_D')

    return output
