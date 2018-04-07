from pycocotools.coco import COCO
from utils import rejection_sample_rec, get_mask_from_diagonal_coord
import numpy as np
import skimage.io as io
import pylab
import signal
import os
import sys

def save_progress(curIndex):
    print("Saving progress...stopping at {}".format(curIndex))
    np.save("curIndex", curIndex)
    sys.exit(0)

def convert_mask_to_rectangular(mask):
    mask_indices = np.where(mask == 1)
    row_indices = mask_indices[0]
    col_indices = mask_indices[1]

    top_left = (np.min(row_indices), np.min(col_indices))
    bottom_right = (np.max(row_indices), np.max(col_indices))

    return get_mask_from_diagonal_coord(top_left, bottom_right, mask)

personMaskDir = "/cs280/home/ubuntu/person_mask"
personDir = "/cs280/home/ubuntu/person"

coco = COCO("/cs280/home/ubuntu/coco/annotations/instances_train2017.json")
personCatId = coco.getCatIds(catNms=["person"])[0]
personIds = coco.getImgIds(catIds=[personCatId])


if not os.path.isfile("curIndex.npy"):
    curIndex = 0
else:
    curIndex = np.load("curIndex.npy")


signal.signal(signal.SIGINT, lambda signum, frame: save_progress(curIndex))

# curIndex = 14417
print("Blacking out images...starting at index {}".format(curIndex))

while curIndex < len(personIds):
    personId = personIds[curIndex]
    personImg = coco.loadImgs(ids=[personId])[0]
    img = io.imread(personImg["coco_url"]) # a numpy ndarray; shape is (H, W, 3)
    annIds = coco.getAnnIds(imgIds=[personId], catIds=[personCatId]) # Get annotations for only people, not other objects
    anns = coco.loadAnns(ids=annIds)
    diag_coord_list = []

    for ann in anns:
        binMask = coco.annToMask(ann) # ndarray; shape is (H, W)
        mask_indices = np.where(binMask == 1)

        row_indices = mask_indices[0]
        col_indices = mask_indices[1]

        top_left = [np.min(row_indices), binMask.shape[1]-np.min(col_indices)]
        bottom_right = [np.max(row_indices), binMask.shape[1]-np.max(col_indices)]

        diag_coord_list.append([top_left, bottom_right])


    # diag_coord_list = [diag_coord_list[0]]
    # print(diag_coord_list)
    new_coords = rejection_sample_rec(im_width=img.shape[0],
                                      im_height=img.shape[1],
                                      min_box_width=img.shape[0]/10,
                                      max_box_width=img.shape[0]/2,
                                      min_box_height=img.shape[1]/10,
                                      max_box_height=img.shape[1]/2,
                                      mask_rec=diag_coord_list,
                                      num_sample=1)
    curIndex += 1
    if len(new_coords) == 0:
        print("Failed to find good box for ", curIndex)
        continue
    else:
        assert len(new_coords) == 1

    for coord in new_coords[0]:
        coord[1] = binMask.shape[1] - coord[1]

    newMask = get_mask_from_diagonal_coord(new_coords[0][0], new_coords[0][1], binMask)

    # cur_mask = np.clip(cur_mask, 0, 1) # in case of overlapping mask
    np.save("{}/{}".format(personMaskDir, personId), newMask)
    io.imsave("{}/{}.jpg".format(personDir, personId), img)
    print("generated person index ", curIndex)
