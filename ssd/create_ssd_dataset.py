from ssd_detector import SSD_Detector
from shutil import copyfile

import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import random

# Percentages of full image width and height.
min_box_width = 0.1
max_box_width = 0.5
min_box_height = 0.1
max_box_height = 0.5

# Only create masks for bounding boxes of at least this confidence.
min_confidence = 0.75

# PASCAL VOC object classes to mask out (15 is person).
filter_classes = [15]

# Source directory to read in images and process them.
IMAGE_SRC_DIR = "/Users/michaelju/cs194/project/examples/"

# Directory to write images to.
IMAGE_DEST_DIR = "/Users/michaelju/cs194/project/ssd_images/"

# Directory to write NumPy masks to.
MASK_DEST_DIR = "/Users/michaelju/cs194/project/ssd_masks/"

# Utility function for processing bounding boxes detected by SSD.
def process_bboxes(rclasses, rscores, rbboxes):
	processed_rbboxes = []

	for i in range(len(rclasses)):
		if rclasses[i] in filter_classes and rscores[i] > min_confidence:
			box_width = rbboxes[i][2] - rbboxes[i][0]
			box_height = rbboxes[i][3] - rbboxes[i][1]

			if box_width >= min_box_width and box_width <= max_box_width and \
			   box_height >= min_box_height and box_height <= max_box_height:
				processed_rbboxes.append(rbboxes[i])

	return processed_rbboxes

# Initialize detector.
ssd_detector = SSD_Detector()

# Make directories.
os.mkdir(IMAGE_DEST_DIR)
os.mkdir(MASK_DEST_DIR)

image_files = glob.glob("{}/*.jpg".format(IMAGE_SRC_DIR))

for image_file in image_files:
	img = plt.imread(image_file)

	if len(img.shape) == 2:
		continue

	rclasses, rscores, rbboxes = ssd_detector.get_bounding_box(img)
	processed_rbboxes = process_bboxes(rclasses, rscores, rbboxes)

	if len(processed_rbboxes) > 0:
		_, file = os.path.split(image_file)
		copyfile(image_file, IMAGE_DEST_DIR + file)

		image_width = img.shape[0]
		image_height = img.shape[1]

		bbox = random.choice(processed_rbboxes)

		top_left = [bbox[0] * image_width, bbox[1] * image_height]
		bottom_right = [bbox[2] * image_width, bbox[3] * image_height]

		# Create mask.
		cols = np.repeat(np.expand_dims(np.arange(img.shape[1]), axis=0), repeats=img.shape[0], axis=0)
		rows = np.repeat(np.expand_dims(np.arange(img.shape[0]), axis=1), repeats=img.shape[1], axis=1)
		newMask = np.logical_and(rows >= top_left[0], rows <= bottom_right[0])
		newMask = np.logical_and(newMask, cols >= top_left[1])
		newMask = np.logical_and(newMask, cols <= bottom_right[1])

		# Save mask.
		np.save(MASK_DEST_DIR + file[:-4] + ".npy", newMask)

		# For visualization.
		# masked_img = (img * (1.0 - np.expand_dims(newMask, 2))).astype('uint8')
		# plt.imshow(masked_img)
		# plt.show()
