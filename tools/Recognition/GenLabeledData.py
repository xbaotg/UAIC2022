import json
import os
from pathlib import Path

import cv2
from tqdm import tqdm

if __name__ == "__main__":
    root_dir = Path("data")
    save_dir = Path("result_train")

    if not save_dir.exists():
        os.mkdir(save_dir)
    if not (save_dir / "images").exists():
        os.mkdir(save_dir / "images")

    images = os.listdir(root_dir / "images")
    output = open(save_dir / "labels.txt", "w+")

    for i in tqdm(range(len(images))):
        fname = images[i]
        path_label = root_dir / "labels" / f"{fname[:-4]}.json"

        if not path_label.exists():
            continue

        words = json.loads("".join(open(path_label, "r").readlines()))
        img = cv2.imread(str(root_dir / "images" / fname))

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
                cv2.imwrite(f"{str(save_dir)}/images/{fname[:-4]}-{idx}.jpg", cropped_img)
                output.write(f"images/{fname[:-4]}-{idx}.jpg\t{label}\n")
            except:
                pass

    output.close()
