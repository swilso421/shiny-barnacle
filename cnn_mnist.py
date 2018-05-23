
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

#This function is the CNN
def cnn_model_fn(features, labels, mode):

    #Input Layer
    input_layer = tf.reshape(features['x'], [-1, 28, 28, 1])

    #First convolutional layer
    conv1 = tf.layers.conv2d(
        inputs = input_layer,
        filters = 32,
        kernel_size = 5,
        padding = 'same',
        activation = tf.nn.relu
    )

    #First pooling layer
    pool1 = tf.layers.max_pooling2d(
        inputs = conv1,
        pool_size = 2,
        strides = 2
    )

    #Second convolutional layer
    conv2 = tf.layers.conv2d(
        inputs = pool1,
        filters = 64,
        kernel_size = 5,
        padding = 'same',
        activation = tf.nn.relu
    )

    #Second pooling layer
    pool2 = tf.layers.max_pooling2d(
        inputs = conv2,
        pool_size = 2,
        strides = 2
    )

    #Flatten the second pooling layer output so that it can go into the dense layer
    pool2_flattened = tf.reshape(pool2, [-1, 7 * 7 * 64])

    #The dense layer will perform a preliminary classification
    dense = tf.layers.dense(
        inputs = pool2_flattened,
        units = 1024,
        activation = tf.nn.relu
    )

    #Dropout regularization so that some of the nodes are deactivated during training passes
    dropout = tf.layers.dropout(
        inputs = dense,
        rate = 0.4,
        training = (mode == tf.estimator.ModeKeys.TRAIN)
    )

    #The output layer, with 10 classes
    logits = tf.layers.dense(
        inputs = dropout,
        units = 10
    )

    #A dictionary containing the predictions made by the network
    predictions = {
        'class': tf.argmax(input = logits, axis = 1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    #If in PREDICT mode, return the prediction made by the CNN
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode = mode, predictions = predictions)

    #Calculate the loss using cross entropy
    loss = tf.lasses.sparse_softmax_cross_entropy(labels = labels, logits = logits)

    #If in TRAIN mode, perform gradient descent to improve the model
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
        train_op = optimizer.minimize(
            loss = loss,
            global_step = tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode = mode, loss = loss, train_op = train_op)

    #If th code gets here, it must have been in EVAL mode, so it returns EVAL results
    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(
            labels = labels,
            predictions = predictions['class']
        )
    }
    return tf.estimator.EstimatorSpec(mode = mode, loss = loss, eval_metric_ops = eval_metric_ops)

def main(unused_argv):

    #Load dataset

    mnist = tf.contrib.learn.datasets.load_dataset('mnist')
    train_data = mnist.train.images
    train_labels = np.asarray(mnist.train.labels, dtype='np.int32')
    eval_data = mnist.test.images
    eval_labels = np.asarray(mnist.test.labels, dtype='np.int32')

if __name__ == "__main__":
    tf.app.run()
