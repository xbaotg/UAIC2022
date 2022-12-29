import numpy as np
import imgaug.augmenters as iaa
import os

from cv2 import cv2
from tqdm import tqdm

seq = iaa.Sequential([
    iaa.AverageBlur(k=(0, 2)),
    iaa.ShearX((-8, 8)),
    iaa.ShearY((-8, 8)),
    iaa.PerspectiveTransform(scale=(0.02, 0.07), keep_size=False),
    iaa.LinearContrast((0.6, 1)),
    iaa.WithPolarWarping(iaa.CropAndPad(percent=(-0.005, 0.005))),
    iaa.pillike.EnhanceSharpness(),
    iaa.pillike.EnhanceContrast(),
    iaa.pillike.EnhanceColor(),
    iaa.WithBrightnessChannels(iaa.Add((-10, 10))),
    iaa.PiecewiseAffine(scale=(0, 0.015)),
    iaa.Add((-10, 10)),
    iaa.PiecewiseAffine(scale=(0, 0.015)),
    iaa.Invert(0.5)
])


# load labels
labels = {}
for line in open("./data_train_after_combine/train/labels.txt").readlines():
    t = line.split("\t")
    fname = t[0][7:]
    labels[fname] = t[1].strip()

images = os.listdir("./data_train_after_combine/train/images/")
output = open("./generated_images/labels.txt", "w+")

to_gen_images = []
cnt = 0

for i in tqdm(range(len(images))):
    fname = images[i]

    if fname in labels:
        img = cv2.imread("./data_train_after_combine/train/images/" + fname, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for idx in range(1, 5):
            img_aug = seq(images=[img])
            cv2.imwrite(f"./generated_images/images/{fname[:-4]}-{idx}.jpg", cv2.cvtColor(img_aug[0], cv2.COLOR_RGB2BGR))

        cv2.imwrite(f"./generated_images/images/{fname[:-4]}-0.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        for i in range(5):
            output.write(f"images/{fname[:-4]}-{i}.jpg\t{labels[fname]}\n")

output.close()
