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

personDir = "/home/ubuntu/person_blacked"
coco = COCO("/home/ubuntu/coco/annotations/instances_train2017.json")
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
    img_blacked = img
    for ann in anns:
        binMask = coco.annToMask(ann) # ndarray; shape is (H, W)
        binMaskInvert = 1 - binMask # swap 1's and 0's in binary mask
        if len(img_blacked.shape) == 2:
            # some images are black and white and don't have a color dimension
            img_blacked = img_blacked * binMaskInvert
        else:
            img_blacked = img_blacked * binMaskInvert[:, :, np.newaxis] # set any pixels in the segmentation mask to black (0, 0, 0)
    io.imsave("{}/{}.jpg".format(personDir, personId), img_blacked)
    curIndex += 1
