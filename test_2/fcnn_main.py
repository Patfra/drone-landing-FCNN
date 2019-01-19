#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
from matplotlib import pyplot as plt
import time
import shutil

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

# --------------------------
# USER-SPECIFIED DATA
# --------------------------

# Tune these parameters

num_classes = 3
NUMBER_OF_CLASSES = 3
image_shape = (352, 640)
IMAGE_SHAPE = (352, 640)
EPOCHS = 40
BATCH_SIZE = 1
DROPOUT = 0.75

# Specify these directory paths

data_dir = './data'
test_data_dir = '../data/360_raw/test'
runs_dir = '../data/runs'
training_dir = './data/data_road/training'
vgg_path = './data/vgg'

# --------------------------
# PLACEHOLDER TENSORS
# --------------------------

correct_label = tf.placeholder(tf.float32, [None, IMAGE_SHAPE[0], IMAGE_SHAPE[1], NUMBER_OF_CLASSES])
learning_rate = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32)



def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # load the model and weights
    model = tf.saved_model.loader.load(sess, ['vgg16'], vgg_path)

    # Get Tensors to be returned from graph
    graph = tf.get_default_graph()
    image_input = graph.get_tensor_by_name('image_input:0')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')
    layer3 = graph.get_tensor_by_name('layer3_out:0')
    layer4 = graph.get_tensor_by_name('layer4_out:0')
    # layer7 = graph.get_tensor_by_name('layer7_out:0') #zamieniam z 7 na 6  layer7_out:0

    for op in graph.get_operations():
        print(str(op.name))

    # TODO: mój własy kod:
    # conv5_1
    parameters = []
    #tf.name_scope = 'conv5_1'
    kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                             stddev=1e-1), name='conv5_1/weights')
    conv = tf.nn.conv2d(layer4, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                         trainable=True, name='conv5_1/biases')
    out = tf.nn.bias_add(conv, biases)
    conv5_1 = tf.nn.relu(out, name='conv5_1')
    parameters += [kernel, biases]

    # conv5_2
    # tf.name_scope = 'conv5_2'
    kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                             stddev=1e-1), name='conv5_2/weights')
    conv = tf.nn.conv2d(conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                         trainable=True, name='conv5_2/biases')
    out = tf.nn.bias_add(conv, biases)
    conv5_2 = tf.nn.relu(out,'conv5_2')
    parameters += [kernel, biases]

    # conv5_3
    # tf.name_scope = 'conv5_3'
    kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                             stddev=1e-1), name='conv5_3/weights')
    conv = tf.nn.conv2d(conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                         trainable=True, name='conv5_3/biases')
    out = tf.nn.bias_add(conv, biases)
    conv5_3 = tf.nn.relu(out, name='conv5_3')
    parameters += [kernel, biases]

    # pool5
    pool5 = tf.nn.max_pool(conv5_3,
                                ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1],
                                padding='SAME',
                                name='pool5')
    layer7 = pool5

    return image_input, keep_prob, layer3, layer4, layer7
#tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # Use a shorter variable name for simplicity
    layer3, layer4, layer7 = vgg_layer3_out, vgg_layer4_out, vgg_layer7_out

    # Apply 1x1 convolution in place of fully connected layer
    fcn8 = tf.layers.conv2d(layer7, filters=num_classes, kernel_size=1, name="fcn8")

    # Upsample fcn8 with size depth=(4096?) to match size of layer 4 so that we can add skip connection with 4th layer
    fcn9 = tf.layers.conv2d_transpose(fcn8, filters=layer4.get_shape().as_list()[-1],
                                      kernel_size=4, strides=(2, 2), padding='SAME', name="fcn9")
    # # TODO: zmiana na fcn32
    # # Upsample again
    # print(layer4.get_shape().as_list()[-1])
    # print(layer3.get_shape().as_list()[-1])
    # # fcn10 = tf.layers.conv2d_transpose(fcn9, filters=layer3.get_shape().as_list()[-1],
    # #                                    kernel_size=4, strides=(2, 2), padding='SAME', name="fcn10_conv2d")
    # # Add a skip connection between current final layer fcn8 and 4th layer
    # fcn9_skip_connected = tf.add(fcn9, layer4, name="fcn9_plus_vgg_layer4")
    #
    # # Upsample again
    # fcn10 = tf.layers.conv2d_transpose(fcn9_skip_connected, filters=layer3.get_shape().as_list()[-1],
    #                                    kernel_size=4, strides=(2, 2), padding='SAME', name="fcn10_conv2d")
    #
    #
    # # Upsample again
    # fcn11 = tf.layers.conv2d_transpose(fcn10, filters=num_classes,
    #                                    kernel_size=16, strides=(8, 8), padding='SAME', name="fcn11")
    # print(layer7.shape.dims)
    # print(fcn8.shape.dims)
    # print(fcn9.shape.dims)
    # print(fcn10.shape.dims)
    # print(fcn11.shape.dims)
    # TODO: fcn8 coś nie działą add - skip connections
    # Add a skip connection between current final layer fcn8 and 4th layer
    fcn9_skip_connected = tf.add(fcn9, layer4, name="fcn9_plus_vgg_layer4")

    # Upsample again
    fcn10 = tf.layers.conv2d_transpose(fcn9_skip_connected, filters=layer3.get_shape().as_list()[-1],
                                       kernel_size=4, strides=(2, 2), padding='SAME', name="fcn10_conv2d")

    # Add skip connection
    fcn10_skip_connected = tf.add(fcn10, layer3, name="fcn10_plus_vgg_layer3")

    # Upsample again
    fcn11 = tf.layers.conv2d_transpose(fcn10_skip_connected, filters=num_classes,
                                       kernel_size=16, strides=(8, 8), padding='SAME', name="fcn11")
    print(fcn11.shape.dims)
    print(layer7.shape.dims)
    print(fcn8.shape.dims)
    print(fcn9.shape.dims)
    print(fcn9_skip_connected.shape.dims)
    print(fcn10.shape.dims)
    print(fcn11.shape.dims)

    return fcn11
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # Reshape 4D tensors to 2D, each row represents a pixel, each column a class
    logits = tf.reshape(nn_last_layer, (-1, num_classes), name="fcn_logits")
    print(nn_last_layer.shape)
    correct_label_reshaped = tf.reshape(correct_label, (-1, num_classes))

    # Calculate distance from actual labels using cross entropy
    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label_reshaped[:])
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                               labels=tf.stop_gradient(correct_label_reshaped[:]))

    # Take mean for total loss
    loss_op = tf.reduce_mean(cross_entropy, name="fcn_loss")

    # The model implements this operation to find the weights/parameters that would yield correct pixel labels
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op, name="fcn_train_op")

    return logits, train_op, loss_op
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    keep_prob_value = 0.5
    learning_rate_value = 0.001
    for epoch in range(epochs):
        # Create function to get batches
        total_loss = 0
        for X_batch, gt_batch in get_batches_fn(batch_size):
            loss, _ = sess.run([cross_entropy_loss, train_op],
                               feed_dict={input_image: X_batch, correct_label: gt_batch,
                                          keep_prob: keep_prob_value, learning_rate: learning_rate_value})

            total_loss += loss

        print("EPOCH {} ...".format(epoch + 1))
        print("Loss = {:.3f}".format(total_loss))
        print()
tests.test_train_nn(train_nn)


def run():
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
    # TODO: tests.test_for_kitti_dataset(data_dir) chyba niepotrzebne??
    # TODO: test bach function - chyba działa
    # get_batches_fn = helper.gen_batch_function('../data', image_shape)
    # for X_batch, gt_batch in get_batches_fn(BATCH_SIZE):
    #     print(gt_batch[0].shape)
    #     plt.figure()
    #     plt.subplot(121)
    #     plt.imshow(X_batch[0])
    #     plt.subplot(122)
    #     plt.imshow(X_batch[0])
    #     plt.show()
    #    input('Pres enter to continue')

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)
    print('Pretrained model downloaded')
    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function('../data', image_shape)
        print('Prepared function to get batches')
        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        with tf.Session() as session:
            # Returns the three layers, keep probability and input layer from the vgg architecture
            image_input, keep_prob, layer3, layer4, layer7 = load_vgg(session, vgg_path)
            print('Get layers form vgg - Done')
            # The resulting network architecture from adding a decoder on top of the given vgg model
            model_output = layers(layer3, layer4, layer7, num_classes)
            print('Added Decoder - Full network architecture On-Line')
            # Returns the output logits, training operation and cost operation to be used
            # - logits: each row represents a pixel, each column a class
            # - train_op: function used to get the right parameters to the model to correctly label the pixels
            # - cross_entropy_loss: function outputting the cost which we are minimizing,
            # lower cost should yield higher accuracy
            logits, train_op, cross_entropy_loss = optimize(model_output, correct_label, learning_rate, num_classes)
            print('Model_optimised')
            print('Initialize all variables - this might slow your computer, just do not worry and wait about 10 min')
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
            # # Initialize all variables
            session.run(tf.global_variables_initializer())
            session.run(tf.local_variables_initializer())
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
            print("Model build successful, starting training")

            # Train the neural network
            train_nn(session, EPOCHS, BATCH_SIZE, get_batches_fn,
                     train_op, cross_entropy_loss, image_input,
                     correct_label, keep_prob, learning_rate)
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
            print('Training Done! :)')
            # Run the model with the test images and save each painted output image (roads painted green)
            helper.save_inference_samples(runs_dir, test_data_dir, session, image_shape, logits, keep_prob, image_input)
            print('Saving model')
            saver = tf.train.Saver()
            output_dir = '../Trained_Model/' + str(time.time())
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            os.makedirs(output_dir)
            saver.save(session, output_dir + '/model.ckpt')
            print("All done!")



if __name__ == '__main__':
    run()
