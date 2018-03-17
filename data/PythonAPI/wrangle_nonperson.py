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

nonPersonDir = "/home/ubuntu/nonperson"
coco = COCO("/home/ubuntu/coco/annotations/instances_train2017.json")
personCatId = coco.getCatIds(catNms=["person"])[0]
personIds = coco.getImgIds(catIds=[personCatId])
allImgIds = coco.getImgIds()
nonPersonIds = list(set(allImgIds) - set(personIds)) # Ids of all images with no people
print(len(nonPersonIds))


if not os.path.isfile("curIndex.npy"):
    curIndex = 0
else:
    curIndex = np.load("curIndex.npy")


signal.signal(signal.SIGINT, lambda signum, frame: save_progress(curIndex))

print("Saving nonperson images...starting at index {}".format(curIndex))
while curIndex < len(nonPersonIds):
    nonPersonId = nonPersonIds[curIndex]
    nonPersonImg = coco.loadImgs(ids=[nonPersonId])[0]
    img = io.imread(nonPersonImg["coco_url"])
    io.imsave("{}/{}.jpg".format(nonPersonDir, nonPersonId), img) # Save this image
    curIndex += 1
