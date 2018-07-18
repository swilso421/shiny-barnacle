#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
from network import *

def train(args):
    print('Train')

def eval(args):
    print('Eval')

def null(args):
    print('Blarg')

def main():

    parser = argparse.ArgumentParser()
    parser.set_defaults(func=null)
    
    subparsers = parser.add_subparsers()
    
    parser_train = subparsers.add_parser('train', help='Train the model from scratch')
    parser_train.add_argument('voxel_directory', help='Directory containing training voxels')
    parser_train.add_argument('-e', '--epochs', type=int, help='Number of training epochs')
    parser_train.add_argument('-c', '--checkpoint', default='3dcnn.ckpt', help='Location to store model checkpoints')
    parser_train.set_defaults(func=train)
    
    parser_eval = subparsers.add_parser('eval', help='

    with tf.Session() as sess:
        
        

if __name__ == '__main__':
    main()
