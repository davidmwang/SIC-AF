import numpy as np

def rejection_sample_rec(im_width,
                         im_height,
                         min_box_width,
                         max_box_width,
                         min_box_height,
                         max_box_height,
                         mask_rec,
                         num_sample):
    """
    randomly sample rectangles within the image
    mask_rec is a list of the diagonal coordinates of the existing masks
    For example:
    [[[1,7],[5,2]],
     [[3,9],[7,4]]]
    represents

    rectangle 1 (mask_rec[0])
    with top left corner at x=1, y=7
    and bottom right corner at x=5, y=2

    rectangle 2 (mask_rec[1])
    with top left corner at x=3, y=9
    and bottom right corner at x=7, y=4
    """
    invalid_rectangles = mask_rec.copy()
    sampled_rectangles = []
    num_attempts = 0
    while num_sample > 0 and num_attempts < 100:
        rand_scales = np.random.uniform(size=(2))
        rand_pt = np.array([min(rand_scales[0]*im_width,im_width-max_box_width),
                            max(rand_scales[1]*im_height,max_box_height)])
        rand_box_w = min_box_width + np.random.uniform() * (max_box_width - min_box_width)
        rand_box_h = min_box_height + np.random.uniform() * (max_box_height - min_box_height)
        new_box = np.array([rand_pt, rand_pt + np.array([rand_box_w, -rand_box_h])])
        valid = True
        for mask in mask_rec:
            valid = valid and not overlap_rec(new_box, mask)
            if not valid: break
        if valid:
            sampled_rectangles.append(new_box)
            num_sample -= 1
        # print(num_sample)
        num_attempts += 1
    return sampled_rectangles

def overlap_rec(rec1, rec2):
    if (rec1[0][0] > rec2[1][0] or rec2[0][0] > rec1[1][0]):
        return False
    if (rec1[0][1] < rec2[1][1] or rec2[0][1] < rec1[1][1]):
        return False
    return True

def visualization(im_width,
                  im_height,
                  min_box_width,
                  max_box_width,
                  min_box_height,
                  max_box_height,
                  mask_rec,
                  num_sample):
    box = rejection_sample_rec(im_width,
                               im_height,
                               min_box_width,
                               max_box_width,
                               min_box_height,
                               max_box_height,
                               mask_rec,
                               num_sample)
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.set_ylim(-5,15)
    ax.set_xlim(-5,15)
    for b in mask_rec:
        ax.add_patch(
            patches.Rectangle(
                (b[0][0], b[1][1]),   # (x,y)
                b[1][0]-b[0][0],      # width
                b[0][1]-b[1][1],      # height
                color='r'
            )
        )
    for b in box:
        ax.add_patch(
            patches.Rectangle(
                (b[0][0], b[1][1]),   # (x,y)
                b[1][0]-b[0][0],      # width
                b[0][1]-b[1][1],      # height
                color='blue'
            )
        )
    plt.show()

def get_mask_from_diagonal_coord(top_left, bottom_right, img):
    """
    Produces a 2-d binary rectangular mask for img given the top left and bottom
    right coordinates of the img.
    """

    cols = np.repeat(np.expand_dims(np.arange(img.shape[1]), axis=0), repeats=img.shape[0], axis=0)
    rows = np.repeat(np.expand_dims(np.arange(img.shape[0]), axis=1), repeats=img.shape[1], axis=1)

    newMask = np.logical_and(rows >= top_left[0], rows <= bottom_right[0])
    newMask = np.logical_and(newMask, cols >= top_left[1])
    newMask = np.logical_and(newMask, cols <= bottom_right[1]).astype(int)

    return newMask
