import os
from pathlib import Path

import imgaug.augmenters as iaa
from cv2 import cv2
from tqdm import tqdm

if __name__ == "__main__":
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

    if not Path("result_aug").exists():
        os.mkdir("result_aug")

    if not Path("result_aug/images").exists():
        os.mkdir("result_aug/images")

    """

    Thay đổi đường dẫn tới thư mục chứa các ảnh và labels.txt cần augment
    | data
        - images
            - im...
            - im ..
        - labels.txt

    """

    root_path = Path("data")
    aug_path = Path("result_aug")

    labels = {}
    labels_data = open(root_path / "labels.txt").readlines()
    images_data = os.listdir(root_path / "images")

    for line in labels_data:
        t = line.split("\t")
        fname = t[0][7:]
        labels[fname] = t[1].strip()

    output = open("result_aug/labels.txt", "w+")

    to_gen_images = []
    cnt = 0

    for i in tqdm(range(len(images_data))):
        fname = images_data[i]

        if fname in labels:
            img = cv2.imread(str(root_path / "images" / fname), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            for idx in range(1, 5):
                img_aug = seq(images=[img])
                cv2.imwrite(str(aug_path / "images" / f"{fname[:-4]}-{idx}.jpg"),
                            cv2.cvtColor(img_aug[0], cv2.COLOR_RGB2BGR))

            cv2.imwrite(str(aug_path / "images" / f"{fname[:-4]}-0.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            for i in range(5):
                output.write(f"images/{fname[:-4]}-{i}.jpg\t{labels[fname]}\n")

    output.close()
