import os
import numpy as np
import ast

from cv2 import cv2
from PIL import ImageFont, ImageDraw, Image

# ---------------------------------------------------

data_path = "../data/uaic2022_public_testB/images/"
outputs = open("./output_rec5.txt").readlines()
labels_from_images = {}
fontpath = "./times.ttf"
font = ImageFont.truetype(fontpath, 20)

for out in outputs:
    t = out.split(" ")
    fname = t[0]
    content = "".join(t[1:-1])[1:-2]

    try:
        conf = float(t[-1][:-2])

        if conf < 0.95:
            continue
    except Exception as e:
        continue

    labels_from_images[fname] = content

# ---------------------------------------------------

outputs = open("./output_rec6.txt").readlines()
labels_from_images_2 = {}

for out in outputs:
    t = out.split(" ")
    fname = t[0]
    content = "".join(t[1:-1])[1:-2]

    try:
        conf = float(t[-1][:-2])

        if conf < 0.95:
            continue
    except Exception as e:
        continue

    labels_from_images_2[fname] = content

# ---------------------------------------------------

labels_locs = {}
for line in open("./labels_gen.txt").readlines():
    t = line.split("\t")
    labels_locs[t[0]] = ast.literal_eval(t[1])

# remove files
for fname in os.listdir("./visualize_data/"):
    os.remove("./visualize_data/" + fname)

# compare two outputs
for org_name in os.listdir(f"{data_path}"):
    img = cv2.imread("{data_path}" + org_name)
    img2 = img.copy()
    changed = False

    for fname in os.listdir("./images_det/"):
        if not fname.startswith(org_name[:-4]):
            continue

        locs = labels_locs[fname]
        locs = [[int(x), int(y)] for x, y in locs]
        xmin = min(t[0] for t in locs)
        xmax = max(t[0] for t in locs)
        ymin = min(t[1] for t in locs)
        ymax = max(t[1] for t in locs)

        if fname not in labels_from_images or fname not in labels_from_images_2 or labels_from_images_2[fname] != labels_from_images[fname]:
            if fname in labels_from_images:
                img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                img_pil = Image.fromarray(img)
                draw = ImageDraw.Draw(img_pil)
                draw.text(tuple([locs[0][0], max(0, locs[0][1] - 30)]), labels_from_images[fname], font = font, fill = (0, 255, 0, 1), stroke_width=1, stroke_fill=(0, 0, 255))
                img = np.array(img_pil)
                changed = True

            if fname in labels_from_images_2:
                img2 = cv2.rectangle(img2, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                img2_pil = Image.fromarray(img2)
                draw = ImageDraw.Draw(img2_pil)
                draw.text(tuple([locs[0][0], max(0, locs[0][1] - 30)]), labels_from_images_2[fname], font = font, fill = (0, 255, 0, 1), stroke_width=1, stroke_fill=(0, 0, 255))
                img2 = np.array(img2_pil)
                changed = True

        else:
            img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)
            img2 = cv2.rectangle(img2, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)

    t = np.hstack((img, img2))
    
    if changed:
        cv2.imwrite("./visualize_data/changed_" + org_name, t)
