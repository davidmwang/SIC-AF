from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import pylab


coco = COCO("/home/ubuntu/coco/annotations/instances_train2017.json")
personCatId = coco.getCatIds(catNms=["person"])[0]
personIds = set(coco.getImgIds(catIds=[personCatId]))
personDir = "/home/ubuntu/person_blacked"
nonPersonDir = "/home/ubuntu/nonperson"

for personId in personIds:
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
allImgIds = set(coco.getImgIds())
nonPersonIds = allImgIds - personIds # Ids of all images with no people
for nonPersonId in nonPersonIds:
    nonPersonImg = coco.loadImgs(ids=[nonPersonId])[0]
    img = io.imread(nonPersonImg["coco_url"])
    io.imsave("{}/{}.jpg".format(nonPersonDir, nonPersonId), img) # Save this image
