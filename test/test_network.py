import numpy as np
import tensorflow as tf
from distutils.version import LooseVersion
import warnings
import helper
import main
import time
import os

hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))

# # Check TensorFlow Version
# assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
# print('TensorFlow Version: {}'.format(tf.__version__))
#
# # Check for a GPU
# if not tf.test.gpu_device_name():
#     warnings.warn('No GPU found. Please use a GPU to train your neural network.')
#     print(tf.test.gpu_device_name())
# else:
#     print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
#
# print(A)
#
#
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
#
#
# # Creates a graph.
# a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
# b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
# c = tf.matmul(a, b)
# # Creates a session with log_device_placement set to True.
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# # Runs the op.
# print(sess.run(c))
#
#
# # Creates a graph.
# with tf.device('/cpu:0'):
#   a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
#   b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
# c = tf.matmul(a, b)
# # Creates a session with log_device_placement set to True.
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# # Runs the op.
# print(sess.run(c))

data_dir = './data'
test_data_dir = '../data/360_raw/test'
runs_dir = '../data/runs'
training_dir = './data/data_road/training'
# TODO: nie zapomnnij zmienić czas/folder/model
#MODEL_FILE_PATH = '../Trained_Model/2019_01_15-19_38_36/model.ckpt'
#MODEL_FILE_PATH = '../Trained_Model/2019_01_15-22_59_50/model.ckpt' #pierwszy działjący -źle
MODEL_FILE_PATH = '../Trained_Model/2019_01_16-00_07_14/model.ckpt' # przetrenowany na 6 obrazkach
image_shape = (352, 640)
IMAGE_SHAPE = (352, 640)
num_classes = 3
NUMBER_OF_CLASSES =3

correct_label = tf.placeholder(tf.float32, [None, IMAGE_SHAPE[0], IMAGE_SHAPE[1], NUMBER_OF_CLASSES])
learning_rate = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32)


# Create function to get batches
#get_batches_fn = helper.gen_batch_function('../data', image_shape)
#print('Prepared function to get batches')
vgg_path = './data\\vgg'

with tf.Session() as session:

    # Returns the three layers, keep probability and input layer from the vgg architecture
    image_input, keep_prob, layer3, layer4, layer7 = main.load_vgg(session, vgg_path)
    print('Get layers form vgg - Done')
    # The resulting network architecture from adding a decoder on top of the given vgg model
    model_output = main.layers(layer3, layer4, layer7, num_classes)
    print('Added Decoder - Full network architecture On-Line')
    # Returns the output logits, training operation and cost operation to be used
    # - logits: each row represents a pixel, each column a class
    # - train_op: function used to get the right parameters to the model to correctly label the pixels
    # - cross_entropy_loss: function outputting the cost which we are minimizing,
    # lower cost should yield higher accuracy
    logits, correct_labels, train_op, cross_entropy_loss = main.optimize(model_output, correct_label, learning_rate,
                                                                    num_classes)
    print('Model_optimised')

    #session.run(tf.global_variables_initializer())
    #session.run(tf.local_variables_initializer())
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
    print("Model build successful, restoring form saved files")
    saver = tf.train.Saver()
    saver.restore(session, MODEL_FILE_PATH)
    print("Model restored successful, starting tests")
    helper.save_inference_samples(runs_dir, test_data_dir, session, image_shape, logits, keep_prob, image_input)
    print("All done!")
