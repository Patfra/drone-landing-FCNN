'''
You should not edit helper.py as part of your submission.

This file is used primarily to download vgg if it has not yet been,
give you the progress of the download, get batches for your training,
as well as around generating and saving the image outputs.
'''

import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm


class DLProgress(tqdm):
	"""
	Report download progress to the terminal.
	:param tqdm: Information fed to the tqdm library to estimate progress.
	"""
	last_block = 0

	def hook(self, block_num=1, block_size=1, total_size=None):
		"""
		Store necessary information for tracking progress.
		:param block_num: current block of the download
		:param block_size: size of current block
		:param total_size: total download size, if known
		"""
		self.total = total_size
		self.update((block_num - self.last_block) * block_size)  # Updates progress
		self.last_block = block_num


def maybe_download_pretrained_vgg(data_dir):
	"""
	Download and extract pretrained vgg model if it doesn't exist
	:param data_dir: Directory to download the model to
	"""
	vgg_filename = 'vgg.zip'
	vgg_path = os.path.join(data_dir, 'vgg')
	vgg_files = [
		os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
		os.path.join(vgg_path, 'variables/variables.index'),
		os.path.join(vgg_path, 'saved_model.pb')]

	missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
	if missing_vgg_files:
		# Clean vgg dir
		if os.path.exists(vgg_path):
			shutil.rmtree(vgg_path)
		os.makedirs(vgg_path)

		# Download vgg
		print('Downloading pre-trained vgg model...')
		with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
			urlretrieve(
				'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
				os.path.join(vgg_path, vgg_filename),
				pbar.hook)

		# Extract vgg
		print('Extracting model...')
		zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
		zip_ref.extractall(data_dir)
		zip_ref.close()

		# Remove zip file to save space
		os.remove(os.path.join(vgg_path, vgg_filename))


def gen_batch_function(data_folder, image_shape):
	"""
	Generate function to create batches of training data
	:param data_folder: Path to folder that contains all the datasets
	:param image_shape: Tuple - Shape of image
	:return:
	"""
	def get_batches_fn(batch_size):
		"""
		Create batches of training data
		:param batch_size: Batch Size
		:return: Batches of training data
		"""
		# Grab image and label paths
		print(data_folder + '/360_raw/*.png')
		image_paths = glob(data_folder + '/360_raw/img_*.png')
		label_paths = glob(data_folder + '/ground_truth/gt_*.png')
		#background_color = np.array([255, 0, 0])
		label_0 = np.array([0, 0, 0])
		label_1 = np.array([128, 128, 128])
		label_2 = np.array([255, 255, 255])

		# Shuffle training data
		#random.shuffle(image_paths)
		print('Number of images:', len(image_paths))
		# Loop through batches and grab images, yielding each batch
		for batch_i in range(0, len(image_paths), batch_size):
			images = []
			gt_images = []
			for image_file in image_paths[batch_i:batch_i+batch_size]:
				gt_image_file =image_file.replace("360_raw\\img", "ground_truth\\gt")
				# Re-size to image_shape
				image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
				gt_image_1 = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)

				# Create "one-hot-like" labels by class
				gt_0 = np.all(gt_image_1 == label_0, axis=2)
				gt_1 = np.all(gt_image_1 == label_1, axis=2)
				gt_2 = np.all(gt_image_1 == label_2, axis=2)

				gt_0 = gt_0.reshape(*gt_0.shape, 1)
				gt_1 = gt_1.reshape(*gt_1.shape, 1)
				gt_2 = gt_2.reshape(*gt_2.shape, 1)

				gt_image = np.concatenate((gt_0, gt_1, gt_2), axis=2)
				# TODO do usunięcia start
				# from matplotlib import pyplot as plt
				# print(gt_image.shape)
				# print(gt_image.dtype)
				# print(gt_image)
				# print(gt_image_1)
				# plt.figure()
				# plt.subplot(131)
				# plt.imshow(gt_image_1)
				# plt.subplot(132)
				# plt.imshow(image)
				# plt.subplot(133)
				# plt.imshow(gt_image*255)
				# plt.show()
				# TODO do usunięcia stop
				images.append(image)
				gt_images.append(gt_image)

			yield np.array(images), np.array(gt_images)
	return get_batches_fn


def gen_test_output(sess, logits, keep_prob, image_pl, data_folder, image_shape):
	"""
	Generate test output using the test images
	:param sess: TF session
	:param logits: TF Tensor for the logits
	:param keep_prob: TF Placeholder for the dropout keep probability
	:param image_pl: TF Placeholder for the image placeholder
	:param data_folder: Path to the folder that contains the datasets
	:param image_shape: Tuple - Shape of image
	:return: Output for for each test image
	"""
	for image_file in glob(data_folder + '/img_*.png'):
		image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

		# Run inference
		im_softmax = sess.run(
			[tf.nn.softmax(logits)],
			{keep_prob: 1.0, image_pl: [image]})
		# Splice out second column (road), reshape output back to image_shape
		a_softmax = np.array(im_softmax)
		maxima=a_softmax.argmax(axis=2)
		classes=maxima.reshape(image_shape[0], image_shape[1], 1)
		segmentation = np.concatenate((classes*127, classes*127, classes*127), axis=2)
		print(os.path.basename(image_file))
		print(a_softmax.shape)
		print(maxima.shape)
		print(classes.shape)
		print(segmentation.shape)
		# Create mask based on segmentation to apply to original image
		mask = scipy.misc.toimage(segmentation, mode="RGB")
		street_im = scipy.misc.toimage(mask)
		#street_im.paste(mask, box=None, mask=mask)

		yield os.path.basename(image_file), np.array(street_im)


def save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image):
	"""
	Save test images with semantic masks of lane predictions to runs_dir.
	:param runs_dir: Directory to save output images
	:param data_dir: Path to the directory that contains the datasets
	:param sess: TF session
	:param image_shape: Tuple - Shape of image
	:param logits: TF Tensor for the logits
	:param keep_prob: TF Placeholder for the dropout keep probability
	:param input_image: TF Placeholder for the image placeholder
	"""
	# Make folder for current run
	output_dir = os.path.join(runs_dir, str(time.strftime("%Y_%m_%d-%H_%M_%S", time.gmtime())))
	if os.path.exists(output_dir):
		shutil.rmtree(output_dir)
	os.makedirs(output_dir)

	# Run NN on test images and save them to HD
	print('Training Finished. Saving test images to: {}'.format(output_dir))
	image_outputs = gen_test_output(
		sess, logits, keep_prob, input_image, data_dir, image_shape)
	for name, image in image_outputs:
		scipy.misc.imsave(output_dir +'/'+ name, image)
