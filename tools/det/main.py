#################################
##
## COMBINE một số ảnh đã được xử lý (change size, aug = roboflow)
## để generate lại bộ ảnh train mới
##
#################################


import os
import sys

sys.path.append("../../")

from cv2 import cv2
from tqdm import tqdm

gened = set()
labels = os.listdir("./labeled/")


for i in tqdm(range(len(labels))):
    label_name = labels[i]
    skip = False
    result = []

    for line in open("./labeled/" + label_name).readlines():
        if line.strip().split(" ")[0] == "1":
            skip = True
            break

        result.append(line)

    if skip:
        continue

    fname = label_name[:-4] + ".jpg"
    img = None

    # cac anh da duoc check + change size
    if os.path.exists("./asdb.v2i.yolov7pytorch/train/images/" + fname):
        img = cv2.imread("./asdb.v2i.yolov7pytorch/train/images/" + fname, cv2.IMREAD_COLOR)
    elif os.path.exists("./train/images/" + fname):
        img = cv2.imread("./train/images/" + fname, cv2.IMREAD_COLOR)

    dh, dw = img.shape[:2]
    new_lines = []

    for line in result:
        _, x, y, w, h = map(float, line.split(' '))

        l = int((x - w / 2) * dw)
        r = int((x + w / 2) * dw)
        t = int((y - h / 2) * dh)
        b = int((y + h / 2) * dh)

        if l < 0:
            l = 0
        if r > dw - 1:
            r = dw - 1
        if t < 0:
            t = 0
        if b > dh - 1:
            b = dh - 1

        # smaller than 3 pixels
        if b - t < 3 or r - l < 3:
            continue

        new_lines.append(line)

    if len(result):
        t = fname.split(".rf.")[0].replace("_jpg", ".jpg")

        if t in gened:
            print(t)
            continue

        gened.add(t)
        cv2.imwrite("./result/images/" + t, img)

        with open("./result/labels/" + label_name.split(".rf.")[0].replace("_jpg", ".txt"), "w+") as f:
            f.writelines(new_lines)
            f.close()
