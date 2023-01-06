import ast
import os
import shutil
import time
import sys
import argparse

import cv2
import torch
import model
import numpy as np

from tqdm import tqdm
from PIL import ImageFont, ImageDraw, Image
from pathlib import Path


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


def predict_text(threshold=0.95, submit=True):
    if not Path("temp-det-images").exists():
        print("Must run generate_det_images()")
        return

    if Path("result").exists():
        shutil.rmtree(Path("result"))
    os.mkdir("result")

    if not submit:
        fontpath = "./times.ttf"
        font = ImageFont.truetype(fontpath, 12)

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
    if submit:
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
    else:
        images = os.listdir("data")
        for i in tqdm(range(len(images))):
            org_name = images[i][:-4]
            img = cv2.imread(f"data/{images[i]}")

            for j, fname in enumerate(img_files):
                if fname.split("-")[0] == org_name:
                    pts = order_points(locs[fname])
                    cv2.rectangle(img, (pts[0][0], pts[0][1]), (pts[2][0], pts[2][1]), (0, 255, 0), 1)
                    img = Image.fromarray(img)
                    draw = ImageDraw.Draw(img)
                    draw.text(tuple([pts[0][0], max(0, pts[0][1] - 30)]), output[j][0], font=font, fill=(0, 255, 0, 1), stroke_width=1, stroke_fill=(0, 0, 255))
                    img = np.array(img)

            cv2.imwrite(f"result/{images[i]}", img)

    os.remove("temp-locs.txt")
    shutil.rmtree("temp-det-images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--task', type=str, default='submit', help='source')
    opt = parser.parse_args()

    if len(os.listdir("models/rec/inference")) < 3:
        print("Execute ./export_ocr_inference.sh first")
        sys.exit(0)

    generate_det_images()

    # sleep 2s to release gpu memory
    time.sleep(2)

    if opt.task == "submit":
        predict_text()
    else:
        predict_text(submit=False)
