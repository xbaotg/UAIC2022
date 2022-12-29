#######################
#
# Compare 2 model yolo
#
#######################

import os
import sys

sys.path.append("../../")

import numpy as np
import model

from cv2 import cv2
from tqdm import tqdm




def visulize(img, locs):
    img = cv2.imread(f"../../../data/uit_public/images/{fname}", cv2.IMREAD_COLOR)

    if locs is None or len(locs) == 0:
        return img

    for idx in range(len(locs)):
        p = locs[idx]

        pts = [
            [p[2], p[0]],
            [p[3], p[0]],
            [p[2], p[1]],
            [p[3], p[1]],
        ]

        xmin = min(p[0] for p in pts)
        xmax = max(p[0] for p in pts)
        ymin = min(p[1] for p in pts)
        ymax = max(p[1] for p in pts)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)

    return img


images = os.listdir("../../../data/uit_public/images/")
d1 = model.DetectWords("../../../weights/sang_22.pt", conf_thres=0.5)
d2 = model.DetectWords("../../../weights/sang_22.pt", conf_thres=0.5, iou_thres=0.4)

for i in tqdm(range(len(images))):
    try:
        fname = images[i]

        img = cv2.imread(f"../../../data/uit_public/images/{fname}", cv2.IMREAD_COLOR)
        img2 = img.copy()

        locs_1 = d1.detect("../../../data/uit_public/images/" + fname, return_locs=True)
        locs_2 = d2.detect("../../../data/uit_public/images/" + fname, return_locs=True)

        changed = len(locs_1) != len(locs_2)

        img_1 = visulize(img, locs_1)
        img_2 = visulize(img2, locs_2)

        cv2.imwrite(f"./result/{'changed-' if changed else ''}{fname}", np.hstack((img_1, img_2)))
    except:
        continue
