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
    np.save("curIndex2", curIndex)
    sys.exit(0)

nonPersonDir = "/cs280/home/ubuntu/no_people"
nonPersonMaskDir = "/cs280/home/ubuntu/no_people_mask"
coco = COCO("/cs280/home/ubuntu/coco/annotations/instances_train2017.json")
personCatId = coco.getCatIds(catNms=["person"])[0]
personIds = coco.getImgIds(catIds=[personCatId])
allImgIds = coco.getImgIds()
nonPersonIds = list(set(allImgIds) - set(personIds)) # Ids of all images with no people


if not os.path.isfile("curIndex2.npy"):
    curIndex = 0
else:
    curIndex = np.load("curIndex2.npy")


signal.signal(signal.SIGINT, lambda signum, frame: save_progress(curIndex))

print("Saving nonperson images...starting at index {}".format(curIndex))
while curIndex < len(nonPersonIds):
    nonPersonId = nonPersonIds[curIndex]
    nonPersonImg = coco.loadImgs(ids=[nonPersonId])[0]
    img = io.imread(nonPersonImg["coco_url"])

    new_coords = rejection_sample_rec(im_width=img.shape[0],
                                      im_height=img.shape[1],
                                      min_box_width=img.shape[0]/10,
                                      max_box_width=img.shape[0]/2,
                                      min_box_height=img.shape[1]/10,
                                      max_box_height=img.shape[1]/2,
                                      mask_rec=[],
                                      num_sample=1)
    curIndex += 1
    if len(new_coords) == 0:
        print("Failed to find good box for ", curIndex)
        continue
    else:
        assert len(new_coords) == 1

    for coord in new_coords[0]:
        coord[1] = img.shape[1] - coord[1]

    newMask = get_mask_from_diagonal_coord(new_coords[0][0], new_coords[0][1], img)
    assert newMask.dtype == np.dtype('bool')

    io.imsave("{}/{}.jpg".format(nonPersonDir, nonPersonId), img) # Save this image
    np.save("{}/{}".format(nonPersonMaskDir, nonPersonId), newMask)
    print("generated nonperson index ... ", curIndex)
