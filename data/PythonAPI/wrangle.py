from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab


coco = COCO()
personCatId = coco.getCatIds(catNms=["person"])[0]
personIds = set(coco.getImgIds(catIds=[personCatId]))

for personId in personIds:
    personImg = coco.loadImgs(ids=[personId])[0]
    img = io.imread(personImg["coco_url"]) # a numpy ndarray
    annId = coco.getAnnIds(imgIds=[personId])[0]
    anns = coco.loadAnns(ids=[annId])
    # Convert each annotation into RLE, merge them?, then convert to binary mask
    # to get  full segmentation mask, then replace those pixels with gray.
    # Finally, save the resulting image.

allImgIds = set(coco.getImgIds())
nonPersonIds = allImgIds - personIds
for nonPersonId in nonPersonIds:
    nonPersonImg = coco.loadImgs(ids=[nonPersonId])[0]
    img = io.imread(nonPersonImg["coco_url"])
    # Save this image
