#! /usr/bin/env python3

import argparse
import cv2
import json
import os

import numpy as np

from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import VGG16, preprocess_input

#Takes in a relative path and formats it as an absolute path with the proper format
def cleanPath(path):
    return os.path.abspath(os.path.expanduser(os.path.normpath(path)))

#Generator function for rendered image file names
def pathGenerator(directory, orbStep = 6, elevStep = 12):
    for elev in range(0, 90, elevStep):
        for orb in range(0, 360, orbStep):
            yield os.path.join(directory, 'elev%3A{};orb%3A{};.png'.format(elev, orb))

#Preprocessing function which expands an image's size with solid color padding
def padImage(image, newShape, paddingType = cv2.BORDER_CONSTANT, color = (64, 64, 64)):

    heightShift = int((newShape[0] - image.shape[0]) / 2)
    widthShift = int((newShape[1] - image.shape[1]) / 2)

    newImage = np.ones((newShape[0], newShape[1], image.shape[2]), dtype = np.uint8)

    newImage = cv2.copyMakeBorder(image, heightShift, heightShift + (image.shape[0] % 2), widthShift, widthShift + (image.shape[1] % 2), paddingType, newImage, color)

    assert(newImage.shape == (224, 224, 3))

    return newImage

#TODO: OpenCV loads images as BGR, so that may have some sort of impact on the filters of the network
def image_loader(directory_path, im_shape = (224, 224)):

    image_list = [padImage(cv2.imread(x), newShape = im_shape) for x in pathGenerator(directory_path)]

    return np.stack(image_list)

#Given a set of feature vectors, calculates the L2 distance between each vector and all other vectors
def cross_compute_distances(data):

    distances = [[np.linalg.norm(row - row2).item() for row2 in data] for row in data]

    return distances

def main():

    #The next 10 lines are the CLI
    def filepath(path):
        try:
            return cleanPath(path)
        except:
            raise argparse.ArgumentTypeError('Value is not a valid path')

    parser = argparse.ArgumentParser()

    parser.add_argument('directory', action='store', type=filepath, help='Directory of image files')

    args = parser.parse_args()



    #TODO: VGG16() has input_tensor and input_shape arguments that could be used instead of manual padding
    #Loads the VGG16 network without the output layer
    model = VGG16(include_top = False)

    #Loads all rendered images of a single model into a tensor
    images = image_loader(args.directory)

    assert(images.shape[0] == 480)

    #Performs the centering and normalization that VGG16 expects
    images = preprocess_input(images.astype('float64'))

    #Calculates the feature vectors, which are just the outputs of the last FC layer
    output = model.predict(images)

    #print(output.shape)

    #Compares each feature vector to all other feature vectors
    #This matrix is graphed as a heatmap to represent the rotational discriminative power of the network
    distances = cross_compute_distances(output)

    #print(type(distances))
    #print(type(distances[0]))
    #print(type(distances[0][0]))

    #Saves distance matrix as json
    with open('cross_distance.json', 'wt+') as f:
        json.dump(distances, f)

if __name__ == '__main__':
    main()
