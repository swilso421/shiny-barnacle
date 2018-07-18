import tensorflow as tf

NUM_CLASSES = 18

def CNN_3D(input_node, stddev = 0.03, name = '3DCNN'):

    filter1 = (3, 3, 3, 1, 32)
    filter2 = (3, 3, 3, 32, 64)
    filter3 = (3, 3, 3, 64, 128)
    filter4 = (3, 3, 3, 128, 256)
    
    weights1 = tf.Variable(tf.truncated_normal(filter1, stddev = stddev), name = name + '_W1')
    weights2 = tf.Variable(tf.truncated_normal(filter2, stddev = stddev), name = name + '_W2')
    weights3 = tf.Variable(tf.truncated_normal(filter3, stddev = stddev), name = name + '_W3')
    weights4 = tf.Variable(tf.truncated_normal(filter4, stddev = stddev), name = name + '_W4')
    weights5 = tf.Variable(tf.truncated_normal([2048, 1024], stddev = stddev), name = name + '_W5')
    weights6 = tf.Variable(tf.truncated_normal([1024, 1024], stddev = stddev), name = name + '_W6')
    weights7 = tf.Variable(tf.truncated_normal([1024, NUM_CLASSES], stddev = stddev), name = name + '_W7')
    
    bias1 = tf.Variable(tf.truncated_normal(32), name = name + '_B1')
    bias2 = tf.Variable(tf.truncated_normal(64), name = name + '_B2')
    bias3 = tf.Variable(tf.truncated_normal(128), name = name + '_B3')
    bias4 = tf.Variable(tf.truncated_normal(256), name = name + '_B4')
    bias5 = tf.Variable(tf.truncated_normal(1024), name = name + '_B5')
    bias6 = tf.Variable(tf.truncated_normal(1024), name = name + '_B6')
    bias7 = tf.Variable(tf.truncated_normal(NUM_CLASSES), name = name + '_B7')
    
    window = [1, 2, 2, 2, 1]
    
    #Convolutional Layer #1
    conv1 = tf.nn.conv3d(input_node,
                         filter = weights1,
                         strides = [1, 1, 1, 1, 1],
                         padding = 'SAME'
                         name = name + '_CONV1')
                         
    conv1 += bias1
    
    relu1 = tf.nn.relu(conv1, name = name + '_RELU1')
    
    pool1 = tf.nn.max_pool3d(relu1,
                             ksize = window,
                             strides = window,
                             padding = 'SAME',
                             name = name + '_POOL1')
    
    
    #Convolutional Layer #2
    conv2 = tf.nn.conv3d(pool1,
                         filter = weights2,
                         strides = [1, 1, 1, 1, 1],
                         padding = 'SAME'
                         name = name + '_CONV2')
                         
    conv2 += bias2
    
    relu2 = tf.nn.relu(conv2, name = name + '_RELU2')
    
    pool2 = tf.nn.max_pool3d(relu2,
                             ksize = window,
                             strides = window,
                             padding = 'SAME',
                             name = name + '_POOL2')
    
    
    #Convolutional Layer #3
    conv3 = tf.nn.conv3d(pool2,
                         filter = weights3,
                         strides = [1, 1, 1, 1, 1],
                         padding = 'SAME'
                         name = name + '_CONV3')
                         
    conv3 += bias3
    
    relu3 = tf.nn.relu(conv3, name = name + '_RELU3')
    
    pool3 = tf.nn.max_pool3d(relu3,
                             ksize = window,
                             strides = window,
                             padding = 'SAME',
                             name = name + '_POOL3')
    
    
    #Convolutional Layer #4
    conv4 = tf.nn.conv3d(pool3,
                         filter = weights4,
                         strides = [1, 1, 1, 1, 1],
                         padding = 'SAME'
                         name = name + '_CONV4')
                         
    conv4 += bias4
    
    relu4 = tf.nn.relu(conv4, name = name + '_RELU4')
    
    pool4 = tf.nn.max_pool3d(relu4,
                             ksize = window,
                             strides = window,
                             padding = 'SAME',
                             name = name + '_POOL4')
                             
    flattened = tf.reshape(pool4, [-1, 2048])
    
    fc5 = tf.nn.matmul(flattened, weights5) + bias5
    
    fc5_relu = tf.nn.relu(fc5, name = name + '_FC5')
    
    if 0.0 < dropout < 1.0:
        fc5_relu = tf.nn.dropout(fc5_relu, dropout, name = name + '_DROP1')
    
    fc6 = tf.nn.matmul(fc5_relu, weights6) + bias6
    
    fc6_relu = tf.nn.relu(fc6, name = name + '_FC6')
    
    if 0.0 < dropout < 1.0:
        fc6_relu = tf.nn.dropout(fc6_relu, dropout, name = name + '_DROP2')
    
    fc7 = tf.nn.matmul(fc6_relu, weights7) + bias7
    
    return fc7
    
