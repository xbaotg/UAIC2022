import sys
import os

from pathlib import Path

sys.path.append("../../")
import model


if __name__ == "__main__":
    r = model.TextRecognizer()
    fnames, pred = r.recognize("data")

    if not Path("result").exists():
        os.mkdir("result")

    for i, fname in enumerate(fnames):

