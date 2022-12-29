import os
import sys

sys.path.append("../../")

import mutils
import json

from cv2 import cv2
from tqdm import tqdm


def gen_images(data_path="../../../data/uit_train/"):
    images = os.listdir(os.path.join(data_path, "images"))
    ftrain = open("./data_train/train/labels.txt", "w+")
    fval = open("./data_train/val/labels.txt", "w+")

    for i in tqdm(range(len(images))):
        fname = images[i]
        path_label = os.path.join(data_path, "labels/" + fname[:-4] + ".json")

        if not os.path.exists(path_label):
            continue

        words = json.loads("".join(open(path_label, "r").readlines()))
        img = cv2.imread(os.path.join(data_path, "images/" + fname))

        for idx, word in enumerate(words):
            pts = word['points']
            label = word['text']

            if label == "#" or label == "###":
                continue

            xmin = int(min(p[0] for p in pts))
            xmax = int(max(p[0] for p in pts))
            ymin = int(min(p[1] for p in pts))
            ymax = int(max(p[1] for p in pts))
            cropped_img = img[ymin:ymax, xmin:xmax]

            try:
                if i < 0.8 * len(images):
                    cv2.imwrite(f"./data_train/train/images/{fname[:-4]}-{idx}.jpg", cropped_img)
                    ftrain.write(f"images/{fname[:-4]}-{idx}.jpg\t{label}\n")
                else:
                    cv2.imwrite(f"./data_train/val/images/{fname[:-4]}-{idx}.jpg", cropped_img)
                    fval.write(f"images/{fname[:-4]}-{idx}.jpg\t{label}\n")
            except:
                pass

    ftrain.close()
    fval.close()


def test_visulize(fname, data_path="../../data/uit_train"):
    img = cv2.imread(os.path.join(data_path, "images/" + fname), cv2.IMREAD_COLOR)
    words = json.loads("".join(open(os.path.join(data_path, f"labels/{fname[:-4]}.json"), "r").readlines()))

    for _, word in enumerate(words):
        pts = word['points']
        label = word['text']

        xmin = min(p[0] for p in pts)
        xmax = max(p[0] for p in pts)
        ymin = min(p[1] for p in pts)
        ymax = max(p[1] for p in pts)

        if label == "###" or label == "#":
            continue

        img = cv2.putText(img, label, (xmin, ymax), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)

    cv2.waitKey(0)


if __name__ == "__main__":
    gen_images()
    # test_visulize("im0004.jpg")
