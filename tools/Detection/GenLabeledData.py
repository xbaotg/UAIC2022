import json
import os
import shutil
import sys

sys.path.append("../../")
import mutils

from cv2 import cv2
from tqdm import tqdm
from pathlib import Path

root_dir = Path("data")
images_data = os.listdir(root_dir / "images")
save_dir = Path("result_det")

if not save_dir.exists():
    os.mkdir(save_dir)
if not (save_dir / "images").exists():
    os.mkdir(save_dir / "images")
if not (save_dir / "labels").exists():
    os.mkdir(save_dir / "labels")

for i in tqdm(range(len(images_data))):
    fname = images_data[i]
    path_label = root_dir / "labels" / f"{fname[:-4]}.json"

    if not path_label.exists():
        continue

    words = json.loads("".join(open(path_label, "r").readlines()))
    img = cv2.imread(str(root_dir / "images" / fname))
    output = open(save_dir / "labels" / fname.replace("jpg", "txt"), "w+")

    for idx, word in enumerate(words):
        pts = word['points']
        label = word['text']

        if label == "#" or label == "###":
            continue

        dw = img.shape[1]
        dh = img.shape[0]
        x, y, w, h = mutils.convert_points_to_yolo((dw, dh), pts)

        shutil.copyfile(root_dir / "images" / fname, save_dir / "images" / fname)
        output.write("0 {} {} {} {}\n".format(x, y, w, h))

    output.close()

# recheck
labels = os.listdir(save_dir / "labels")
for i in range(len(labels)):
    fname = labels[i]

    if not Path(save_dir / "images" / fname.replace("txt", "jpg")).exists():
        print("removing: " + fname)
        os.remove(Path(save_dir / "labels" / fname))
