
import tensorflow as tf
import sapi as s

#Width and height to scale images to for the network
GBL_IM_SCALE_W = 64
GBL_IM_SCALE_H = 64

#Number of filters in each convolutional layer
GBL_FILTERS_1 = 16
GBL_FILTERS_2 = 32

#Dropout keep rate during training
GBL_DROPOUT = 0.5

#Number of nodes in the dense layer
GBL_FC_UNITS = 1024

#Loads an image file as a tensor. Supports bmp, png, jpeg, and gif
#Recommended use case is with dataset.map
def image_loader(path, channels = 0, shape = None, name = None):
    '''Loads an image file as a tensor'''

    #Read the raw binary data
    raw_image = tf.readfile(path)

    #Decode the binary data into a tensor
    image = tf.image.decode_image(raw_image, channels, name)

    #Scales images to a common size
    scaled_image = tf.image.resize_images(image, [GBL_IM_SCALE_W, GBL_IM_SCALE_H])

    return scaled_image

#Create the network used for the XRAY task
def XRAYnetwork(input_node, mode):

    conv1 = s.conv_layer(input_node, 1, GBL_FILTERS_1, [4, 4], name = 'CONV1')

    pool1 = s.pool_layer(conv1)

    conv2 = s.conv_layer(pool1, GBL_FILTERS_1, GBL_FILTERS_2, [4, 4], name = 'CONV2')

    pool2 = s.pool_layer(conv2)

    pool2_flat = tf.reshape(pool2, [-1, GBL_FILTERS_2 * GBL_IM_SCALE_W * GBL_IM_SCALE_H / 16])

    dense = s.fc_layer(pool2_flat, GBL_FC_UNITS, dropout = GBL_DROPOUT if mode == 'train' else 1.0, name = 'FC1')

    output = s.fc_layer(dense, 2, relu = False, name = 'OUTPUT')

    return output

def main():
    print('Hello World!')

    with tf.Session() as sess:

        output = XRAYnetwork(data, mode)

        loss = tf.nn.softmax_cross_entropy_with_logits_v2(output, labels)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate = GBL_LEARNING_RATE).minimize(loss)

if __name__ == '__main__':
    main()
