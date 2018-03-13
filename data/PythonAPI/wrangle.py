from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
#import matplotlib.pyplot as plt
import pylab


coco = COCO("/home/ubuntu/coco/annotations/instances_val2017.json")
personCatId = coco.getCatIds(catNms=["person"])[0]
personIds = set(coco.getImgIds(catIds=[personCatId]))
i = 0

for personId in personIds:
    personImg = coco.loadImgs(ids=[personId])[0]
    img = io.imread(personImg["coco_url"]) # a numpy ndarray; shape is (H, W, 3)
    annId = coco.getAnnIds(imgIds=[personId])[0]
    anns = coco.loadAnns(ids=[annId])
    assert(len(anns) == 1)
    binMask = coco.annToMask(anns[0]) # ndarray; shape is (H, W)
    binMaskInvert = 1 - binMask # swap 1's and 0's in binary mask
    img_blacked = img * binMaskInvert[:, :, np.newaxis] # set any pixels in the segmentation mask to black (0, 0, 0)
    io.imsave("/home/ubuntu/coco/test_img_{}.jpg".format(personId), img)
    io.imsave("/home/ubuntu/coco/test_img_{}_blacked.jpg".format(personId), img_blacked)
    i += 1
    if i == 10:
        break
    # Convert each annotation into RLE, merge them?, then convert to binary mask
    # to get  full segmentation mask, then replace those pixels with gray.
    # Finally, save the resulting image.

"""
allImgIds = set(coco.getImgIds())
nonPersonIds = allImgIds - personIds
for nonPersonId in nonPersonIds:
    nonPersonImg = coco.loadImgs(ids=[nonPersonId])[0]
    img = io.imread(nonPersonImg["coco_url"])
    # Save this image
"""
