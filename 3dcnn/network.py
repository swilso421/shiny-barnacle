import tensorflow as tf

def CNN_3D(input_node, outputSize = 256, volDropout = 0.8, fcDropout = 0.5, stddev = 0.03, name = '3DCNN'):

    assert(outputSize <= 512)
    
    with tf.name_scope(name) as scope:
        
        block1 = NINBlock(input_node, (1, 48, 6), dropout = volDropout, stddev = stddev, name = name + '_NIN1')
        
        block2 = NINBlock(block1, (48, 96, 5), dropout = volDropout, stddev = stddev, name = name + '_NIN2')
        
        block3 = NINBlock(block2, (96, 512, 3), dropout = volDropout, stddev = stddev, name = name + '_NIN3')
        
        flattened = tf.reshape(block3, [-1, 4096])
        
        with tf.name_scope(name + '_FC1') as scope:
            
            weights1 = tf.Variable(tf.truncated_normal([4096, 512], stddev = stddev), name = name + '_FC1_W')
            bias1 = tf.Variable(tf.truncated_normal(512, stddev = stddev), name = name + '_FC1_B')
        
            fc1 = tf.nn.matmul(flattened, weights1, name = name + '_FC1_MATMUL') + bias1
            
            fc1_relu = tf.nn.relu(fc1, name = name + '_FC1_RELU')
            
            if 0.0 < dropout < 1.0:
                
                fc1_relu = tf.nn.dropout(fc1_relu, fcDropout, name = name + '_FC1_DROP')
            
        with tf.name_scope(name + '_FC2') as scope:
            
            weights2 = tf.Variable(tf.truncated_normal([512, outputSize], stddev = stddev), name = name + '_FC2_W')
            bias2 = tf.Variable(tf.truncated_normal(outputSize, stddev = stddev), name = name + '_FC2_B')
            
            fc2 = tf.nn.matmul(fc1_relu, weights2, name = name + '_FC2_MATMUL') + bias2
        
        return fc2

def NINBlock(input_node, params = (1, 48, 6), dropout = 0.8, stddev = 0.03, name = 'NIN'):
    
    with tf.name_scope(name) as scope:
        
        filter1 = (params[2], params[2], params[2], params[0], params[1])
        filter2 = (1, 1, 1, params[1], params[1])
        
        vol1 = VolumetricLayer(input_node, filter1, stddev = stddev, name = name + '_VOL1')
        
        vol2 = VolumetricLayer(vol1, filter2, stddev = stddev, name = name + '_VOL2')
        
        vol3 = VolumetricLayer(vol2, filter2, stddev = stddev, name = name + '_VOL3')
        
        if 0.0 < dropout < 1.0:
        
            #Apply dropout with a keep probability of <dropout>
            return tf.nn.dropout(vol3, dropout, name = name + '_VDROP')
            
        else:
        
            return vol3
    
def VolumetricLayer(input_node, filter, stddev = 0.03, useBatchNorm = True, useReLU = True, name = 'VOL'):

    with tf.name_scope(name) as scope:
        
        weights = tf.Variable(tf.truncated_normal(filter, stddev = stddev), name = name + '_W')
        bias = tf.Variable(tf.truncated_normal(filter[4], stddev = stddev), name = name + '_B')
        
        #Apply volumetric convolution
        output =  tf.nn.conv3d(input_node,
                             filter = weights,
                             strides = [1, 1, 1, 1, 1],
                             padding = 'SAME',
                             name = name + '_CONV')
        
        if useBatchNorm:
            
            #XXX: Need to figure out the correct value for the 'axes' argument
            mean, var = tf.nn.moments(output, axes = [0], name = name + '_MOM')
            
            #Perform batch normalization
            output = tf.nn.batchnormalization(output, mean, var, offset = None, scale = None, name = name + '_BN')
            
        if useReLU:
            
            #Apply ReLU activation function to tensor
            output = tf.nn.relu(output, name = name + '_RELU')
        
        return output
