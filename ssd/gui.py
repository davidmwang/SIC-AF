import matplotlib.pyplot as plt
import matplotlib.patches as patches
import glob
import tensorflow as tf
import numpy as np
import argparse
from ssd_detector import SSD_Detector
from enum import Enum
from PIL import Image

class mode(Enum):
    IDLE = 0
    BBOX_SELECTION = 1

# Only create masks for bounding boxes of at least this confidence.
min_confidence = 0.50

# PASCAL VOC object classes to mask out (15 is person).
filter_classes = [15]

def process_bboxes(rclasses, rscores, rbboxes, category=15):
    processed_rbboxes = []

    for i in range(len(rclasses)):
        if rclasses[i] == category and rscores[i] > min_confidence:
            processed_rbboxes.append(rbboxes[i])

    return processed_rbboxes

def compute_bounding_box(img, category=15):
    img_width = img.shape[1]
    img_height = img.shape[0]

    ssd_detector = SSD_Detector()
    rclasses, rscores, rbboxes = ssd_detector.get_bounding_box(img)

    processed_rbboxes = process_bboxes(rclasses, rscores, rbboxes, category)

    # Consists of bottom left coordinates.
    output = []

    for box in processed_rbboxes:
        box_width = img_width * (box[3] - box[1])
        box_height = img_height * (box[2] - box[0])

        bottom_left_x = img_width * box[1]
        bottom_left_y = img_height * box[0]

        output.append([bottom_left_x, bottom_left_y, box_width, box_height])

    return output

def window(img_path):
    displayed_img = plt.imread(img_path)
    bbox = compute_bounding_box(displayed_img)

    m = mode.BBOX_SELECTION
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.imshow(displayed_img)

    while True:
        plt.ginput(1, timeout=-1, show_clicks=False)
        if m == mode.BBOX_SELECTION:
            for box in bbox:
                bottom_left_x, bottom_left_y, box_width, box_height = box

                ax.add_patch(patches.Rectangle((bottom_left_x,bottom_left_y),
                                               box_width,box_height,
                                               linewidth=1,
                                               edgecolor='g',
                                               facecolor='none'))
            plt.imshow(displayed_img)
            loc = plt.ginput(1, timeout=-1, show_clicks=False)
            loc = np.asarray(loc, dtype=np.int32)[0]
            print(loc)
            candidate_box = None
            for box in bbox:
                bottom_left_x, bottom_left_y, box_width, box_height = box
                if loc[0] >= bottom_left_x and loc[0] <= bottom_left_x + box_width and \
                   loc[1] >= bottom_left_y and loc[1] <= bottom_left_y + box_height:
                   candidate_box = box
            displayed_img[int(bottom_left_y):int(bottom_left_y + box_height),
                          int(bottom_left_x):int(bottom_left_x+ box_width)] = 0
            ax.imshow(displayed_img)
            # find corresponding box
            # PROCESSING


def main():
    parser = argparse.ArgumentParser(description='Reading image')
    parser.add_argument('--path', type=str, help='path to image')
    parsed = parser.parse_args()
    window(parsed.path)

if __name__ == "__main__":
    main()
