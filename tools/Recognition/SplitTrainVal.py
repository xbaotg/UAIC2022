import os
import shutil
from pathlib import Path

from cv2 import cv2
from tqdm import tqdm

if __name__ == "__main__":
    root_dir = Path("data")
    save_dir = Path("result_train_val")

    if not save_dir.exists():
        os.mkdir(save_dir)
    if not (save_dir / "train").exists():
        os.mkdir(save_dir / "train")
    if not (save_dir / "val").exists():
        os.mkdir(save_dir / "val")
    if not (save_dir / "train" / "images").exists():
        os.mkdir(save_dir / "train" / "images")
    if not (save_dir / "val" / "images").exists():
        os.mkdir(save_dir / "val" / "images")

    images = os.listdir(root_dir / "images")
    ftrain = open(save_dir / "train/labels.txt", "w+")
    fval = open(save_dir / "val/labels.txt", "w+")
    input = open(root_dir / "labels.txt", "r+").readlines()
    output = open(root_dir / "labels.txt", "w+")

    labels = {}
    for line in input:
        labels[line.split("\t")[0].split("/")[-1]] = line.split("\t")[1].strip()

    for i in tqdm(range(len(images))):
        fname = images[i]
        img = cv2.imread(str(root_dir / "images" / fname))
        label = labels[fname]

        try:
            if i < 0.8 * len(images):
                shutil.copy(root_dir / "images" / fname, save_dir / "train" / "images" / fname)
                ftrain.write(f"images/{fname}\t{label}\n")
            else:
                shutil.copy(root_dir / "images" / fname, save_dir / "val" / "images" / fname)
                fval.write(f"images/{fname}\t{label}\n")
        except:
            print("skip: " + fname)

    ftrain.close()
    fval.close()
