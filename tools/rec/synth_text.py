import numpy as np
import os
import random as rd

from tqdm import tqdm
from cv2 import cv2
from trdg.generators import (
    GeneratorFromDict,
)

fonts = list(map(lambda x : f"./fonts/{x}", os.listdir("./fonts/")))
output = open("./synth_text_images/labels.txt", "w+")

for i in tqdm(range(100)):
    generator = GeneratorFromDict(
        blur=rd.randint(0, 2),
        size=rd.randint(20, 60),
        random_blur=True,
        skewing_angle=35,
        random_skew=True,
        path="./dict_gen.txt",
        language="vi",
        fonts=fonts,
        background_type=rd.randint(0, 2),
        text_color="#ff0000,#0000ff",
        distorsion_type=rd.randint(0, 1),
        fit=bool(rd.randint(0, 1))
    )

    cnt = 0
    for img, lbl in generator:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"./synth_text_images/images/syn-{i}-{cnt}.jpg", img)
        output.write(f"images/syn-{i}-{cnt}.jpg\t{lbl}\n")
        
        if cnt == 1000:
            break

        cnt += 1
        next(generator)

output.close()
