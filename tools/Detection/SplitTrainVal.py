import os
import shutil
from pathlib import Path

import cv2
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
    if not (save_dir / "train" / "labels").exists():
        os.mkdir(save_dir / "train" / "labels")
    if not (save_dir / "val" / "labels").exists():
        os.mkdir(save_dir / "val" / "labels")

    images = os.listdir(root_dir / "images")

    for i in tqdm(range(len(images))):
        fname = images[i]
        img = cv2.imread(str(root_dir / "images" / fname))

        try:
            if i < 0.8 * len(images):
                shutil.copy(root_dir / "images" / fname, save_dir / "train" / "images" / fname)
                shutil.copy(root_dir / "labels" / fname.replace("jpg", "txt"),
                            save_dir / "train" / "labels" / fname.replace("jpg", "txt"))
            else:
                shutil.copy(root_dir / "images" / fname, save_dir / "val" / "images" / fname)
                shutil.copy(root_dir / "labels" / fname.replace("jpg", "txt"),
                            save_dir / "val" / "labels" / fname.replace("jpg", "txt"))
        except:
            print("skip: " + fname)
