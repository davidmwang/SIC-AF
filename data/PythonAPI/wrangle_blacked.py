from pycocotools.coco import COCO
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

    cols = np.repeat(np.expand_dims(np.arange(mask.shape[1]), axis=0), repeats=mask.shape[0], axis=0)
    rows = np.repeat(np.expand_dims(np.arange(mask.shape[0]), axis=1), repeats=mask.shape[1], axis=1)

    newMask = np.logical_and(rows >= top_left[0], rows <= bottom_right[0])
    newMask = np.logical_and(newMask, cols >= top_left[1])
    newMask = np.logical_and(newMask, cols <= bottom_right[1]).astype(int)

    return newMask

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

print("Blacking out images...starting at index {}".format(curIndex))
while curIndex < len(personIds):
    personId = personIds[curIndex]
    personImg = coco.loadImgs(ids=[personId])[0]
    img = io.imread(personImg["coco_url"]) # a numpy ndarray; shape is (H, W, 3)
    annIds = coco.getAnnIds(imgIds=[personId], catIds=[personCatId]) # Get annotations for only people, not other objects
    anns = coco.loadAnns(ids=annIds)
    cur_mask = np.zeros((img.shape[0], img.shape[1]))

    for ann in anns:
        binMask = coco.annToMask(ann) # ndarray; shape is (H, W)
        cur_mask += convert_mask_to_rectangular(binMask)

    cur_mask = np.clip(cur_mask, 0, 1) # in case of overlapping mask
    np.save("{}/{}".format(personMaskDir, personId), cur_mask)
    io.imsave("{}/{}.jpg".format(personDir, personId), img)
    
    curIndex += 1
