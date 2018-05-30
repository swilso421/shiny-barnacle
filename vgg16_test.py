#! /usr/bin/env python3

import argparse
import cv2
import json
import os

import numpy as np

from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import VGG16, preprocess_input

def cleanPath(path):
    return os.path.abspath(os.path.expanduser(os.path.normpath(path)))

def pathGenerator(directory, orbStep = 6, elevStep = 12):
    for elev in range(0, 90, elevStep):
        for orb in range(0, 360, orbStep):
            yield cleanPath(os.path.join(directory, 'elev:{};orb:{};.png'.format(elev, orb)))

def padImage(image, newShape, paddingType = cv2.BORDER_CONSTANT, color = (64, 64, 64)):

    heightShift = int((newShape[0] - image.shape[0]) / 2)
    widthShift = int((newShape[1] - image.shape[1]) / 2)

    newImage = cv2.copyMakeBorder(image, heightShift, heightShift, widthShift, widthShift, paddingType, newImage, color)

    return newImage

#TODO: OpenCV loads images as BGR, so that may have some sort of impact on the filters of the network
def image_loader(directory_path, shape = (224, 224)):

    image_list = [padImage(cv2.imread(x), shape) for x in pathGenerator(directory_path)]

    return np.stack(image_list)

def cross_compute_distances(data):

    distances = [[np.linalg.norm(row - row2) for row2 in data] for row in data]

    return distances

def main():

    def filepath(path):
        try:
            return cleanPath(path)
        except:
            raise argparse.ArgumentTypeError('Value is not a valid path')

    parser = argparse.ArgumentParser()

    parser.add_argument('directory', action='store', type=filepath, help='Directory of image files')

    args = pargers.parse_args()

    #TODO: VGG16() has input_tensor and input_shape arguments that could be used instead of manual padding
    model = VGG16(include_top = False)

    images = image_loader(args.directory)

    assert(images.shape[0] == 480)

    images = preprocess_input(images)

    output = model.predict(images)

    distances = cross_compute_distances(output)

    with open('cross_distance.json', 'wt+') as f:
        json.dump(distances, f)

if __name__ == '__main__':
    main()
