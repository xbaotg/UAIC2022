import os
import random as rd
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from trdg.generators import GeneratorFromDict

if __name__ == "__main__":
    if not Path("result_synth").exists():
        os.mkdir("result_synth")

    if not Path("result_synth/images").exists():
        os.mkdir("result_synth/images")

    fonts = list(map(lambda x: f"configs/fonts/{x}", os.listdir("configs/fonts/")))
    output = open("result_synth/labels.txt", "w+")

    for i in tqdm(range(50)):
        generator = GeneratorFromDict(
            size=rd.randint(25, 60),
            skewing_angle=10,
            random_skew=True,
            path="configs/dict_gen.txt",
            language="vi",
            fonts=fonts,
            background_type=rd.randint(0, 2),
            text_color="#000000,#888888",
            fit=True,
        )

        cnt = 0
        for img, lbl in generator:
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"./result_synth/images/syn-{i}-{cnt}.jpg", img)
            output.write(f"images/syn-{i}-{cnt}.jpg\t{lbl}\n")

            if cnt == 1000:
                break

            cnt += 1
            next(generator)

    output.close()
