import numpy as np
import imgaug.augmenters as iaa
import os
import sys

sys.path.append("../../")

import mutils
from cv2 import cv2
from tqdm import tqdm

from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


seq = iaa.SomeOf(8, [
    iaa.AverageBlur(k=(0, 2)),
    iaa.ShearX((-20, 20)),
    iaa.ShearY((-20, 20)),
    iaa.PerspectiveTransform(scale=(0.02, 0.08), keep_size=False),
    iaa.LinearContrast((0.75, 1.5)),
    iaa.WithPolarWarping(iaa.CropAndPad(percent=(-0.01, 0.01))),
    iaa.pillike.EnhanceSharpness(),
    iaa.pillike.EnhanceContrast(),
    iaa.pillike.EnhanceColor(),
    iaa.WithBrightnessChannels(iaa.Add((-20, 20))),
    iaa.AddToBrightness((-30, 30)),
    iaa.AddToBrightness((-20, 20)),
    iaa.Dropout(p=(0, 0.1)),
    iaa.Add((-10, 10)),
    # iaa.Invert(0.5),
    # iaa.Rot90(1),
    iaa.Rot90(2),
    # iaa.Rot90(3),
])

def convert_bb_yolo(b, w, h):
    result = []

    for bb in b:
        pts = [
            bb[0],
            bb[1]
        ]

        result.append(mutils.convert_points_to_yolo((w, h), pts))

    return result


images = os.listdir("./result/train/images/")
for i in tqdm(range(len(images))):
    fname = images[i]

    try:
        img = cv2.imread(f"./result/train/images/{fname}", cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        bbox_org = []
        for line in open(f"./result/train/labels/{fname.replace('jpg', 'txt')}"):
            _, x, y, w, h = map(float, line.split(" "))
            pts = mutils.convert_yolo_to_points(x, y, w, h, img.shape[1], img.shape[0])
            bbox_org.append(BoundingBox(x1=pts[0][0], y1=pts[0][1], x2=pts[1][0], y2=pts[1][1]))

        bbox_org = BoundingBoxesOnImage(bbox_org, shape=img.shape)

        for idx in range(1, 5):
            img_aug, bbox_aug = seq(images=[img], bounding_boxes=bbox_org)
            bbox_aug = bbox_aug.remove_out_of_image().clip_out_of_image()
            output = open(f"./generated_images/labels/{fname[:-4]}-{idx}.txt", "w+")
            dh, dw = img_aug[0].shape[:2]

            cv2.imwrite(f"./generated_images/images/{fname[:-4]}-{idx}.jpg", cv2.cvtColor(img_aug[0], cv2.COLOR_RGB2BGR))
            for (x, y, w, h) in convert_bb_yolo(bbox_aug, dw, dh):
                output.write(f"0 {x} {y} {w} {h}\n")

            output.close()

        output = open(f"./generated_images/labels/{fname[:-4]}-0.txt", "w+")
        dh, dw = img.shape[:2]

        cv2.imwrite(f"./generated_images/images/{fname[:-4]}-0.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        for (x, y, w, h) in convert_bb_yolo(bbox_org, dw, dh):
            output.write(f"0 {x} {y} {w} {h}\n")

        output.close()
    except Exception as e:
        print(fname, e)
