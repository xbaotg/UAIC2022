import os
import sys
import ast

from cv2 import cv2
from tqdm import tqdm

sys.path.append("../")

import model


def order_points(pts):
    xmin = min(p[0] for p in pts)
    xmax = max(p[0] for p in pts)
    ymin = min(p[1] for p in pts)
    ymax = max(p[1] for p in pts)

    return [
        [xmin, ymin],
        [xmax, ymin],
        [xmax, ymax],
        [xmin, ymax]
    ]


def gen(path_folder="../data/uit_train/images/"):
    d = model.DetectWords("../weights/sang_22.pt", conf_thres=0.5, iou_thres=0.5)
    d2 = model.DetectWords("../weights/yolo_887.pt", conf_thres=0.5, iou_thres=0.5)

    output = open("./labels_gen.txt", "w+")
    images = os.listdir(path_folder)

    # remove old
    for fname in os.listdir("./images_det/"):
        os.remove("./images_det/" + fname)

    for i in tqdm(range(len(images))):
        fname = images[i]
        locs = d.detect(path_folder + fname, return_locs=True)  # conf = 0.5

        if locs is None:
            locs = d2.detect(path_folder + fname, return_locs=True)  # conf = 0.5

            if locs is None:
                print("none", fname)
                continue

        img = cv2.imread(path_folder + fname, cv2.IMREAD_COLOR)

        for idx, p in enumerate(locs):
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

            name_save = fname[:-4] + "-" + str(idx) + ".jpg"
            output.write(name_save + "\t" + str(pts) + "\n")
            cv2.imwrite("./images_det/" + name_save, img[ymin:ymax, xmin:xmax])

    output.close()


def gen_result(threshold=0.95):
    outputs = open("./output.txt").readlines()
    locs = {}

    for line in open("./labels_gen.txt", "r+").readlines():
        t = line.split("\t")
        locs[t[0]] = ast.literal_eval(t[1])

    # remove old files
    for fname in os.listdir("./result/"):
        os.remove("./result/" + fname)

    # process
    for line in outputs:
        t = line.split(" ")
        fname = t[0]
        content = " ".join(t[1:-1])[1:-2]
        conf = float(t[-1][:-2])

        if conf < threshold or len(content) == 0:
            continue

        try:
            pts = order_points(locs[fname])
        except:
            continue

        loc = ",".join(f"{int(x)},{int(y)}" for (x, y) in pts)
        org_name = fname.split("-")[0]

        with open("./result/" + org_name + ".txt", "a+") as f:
            f.write(loc + "," + content + "\n")


if __name__ == "__main__":
    gen("../data/uaic2022_public_testB/images/")
    # gen_result(0.95)
