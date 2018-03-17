import os
import errno

blacked = "/home/ubuntu/person_blacked"
with open("./list_all_blacked.txt", "r") as all_blacked:
    for img_id in all_blacked:
        img_id = img_id.strip("\n")
        try:
            os.remove("{}/{}.jpg".format(blacked, img_id))
        except OSError as e:
            if e.errno != errno.ENOENT:
                raise
            else:
                print("{}/{}.jpg does not exist".format(blacked, img_id))
