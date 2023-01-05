import ast
import os
import shutil
import time
from pathlib import Path

import torch
from cv2 import cv2
from tqdm import tqdm

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


def generate_det_images():
    # load models
    d = model.DetectWords("models/det/yolo.pt", conf_thres=0.43, iou_thres=0.5, imgsz=960)
    d2 = model.DetectWords("models/det/yolo.pt", conf_thres=0.3, iou_thres=0.5, imgsz=960)

    # processing existed folders
    if Path("temp-det-images").exists():
        shutil.rmtree(Path("temp-det-images"))
    os.mkdir(Path("temp-det-images"))

    # Process
    print("")
    print("-----------------------------------------")
    print("Generating words ...")

    output = open("temp-locs.txt", "w+")
    images = os.listdir("data")

    for i in tqdm(range(len(images))):
        fname = images[i]

        if ".jpg" not in fname:
            continue

        locs, img = d.detect(f"data/{fname}")

        if locs is None:
            locs, img = d2.detect(f"data/{fname}")
            if locs is None:
                continue

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

            name_save = f"{fname[:-4]}-{idx}.jpg"

            output.write(name_save + "\t" + str(pts) + "\n")
            cv2.imwrite(f"temp-det-images/{name_save}", img[ymin:ymax, xmin:xmax])

    output.close()

    # release gpu memory
    with torch.no_grad():
        torch.cuda.empty_cache()


def predict_text(threshold=0.95):
    if not Path("temp-det-images").exists():
        print("Must run generate_det_images()")
        return

    if Path("result").exists():
        shutil.rmtree(Path("result"))
    os.mkdir("result")

    # Process
    locs = {}
    for line in open("temp-locs.txt", "r+").readlines():
        t = line.split("\t")
        locs[t[0]] = ast.literal_eval(t[1])

    print("Predicting ...")
    rec = model.TextRecognizer()
    img_files, output = rec.recognize("temp-det-images")
    img_files = [t.split("/")[-1] for t in img_files]

    print("Generating result ...")
    for idx, fname in enumerate(img_files):
        pred = output[idx][0]
        conf = output[idx][1]

        if conf < threshold:
            continue

        pts = order_points(locs[fname])
        loc = ",".join(f"{int(x)},{int(y)}" for (x, y) in pts)
        org_name = fname.split("-")[0]

        with open("result/" + org_name + ".txt", "a+") as f:
            f.write(f"{loc},{pred}\n")
            f.close()

    os.remove("temp-locs.txt")
    shutil.rmtree("temp-det-images")


if __name__ == "__main__":
    generate_det_images()

    # sleep 5s to release gpu memory
    time.sleep(5)

    predict_text()
