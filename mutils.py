import json

import cv2
import numpy as np


def convert_yolo_to_points(x, y, w, h, dw, dh):
    l = int((x - w / 2) * dw)
    r = int((x + w / 2) * dw)
    t = int((y - h / 2) * dh)
    b = int((y + h / 2) * dh)

    if l < 0:
        l = 0
    if r > dw - 1:
        r = dw - 1
    if t < 0:
        t = 0
    if b > dh - 1:
        b = dh - 1

    return [(l, t), (r, b)]


def convert_points_to_yolo(size, points):
    x_min = min([p[0] for p in points])
    x_max = max([p[0] for p in points])
    y_min = min([p[1] for p in points])
    y_max = max([p[1] for p in points])

    box = (x_min, x_max, y_min, y_max)

    dw = 1. / size[0]
    dh = 1. / size[1]

    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0

    w = box[1] - box[0]
    h = box[3] - box[2]

    x = x * dw
    w = w * dw

    y = y * dh
    h = h * dh

    return (x, y, w, h)


def visualize_from_labels(s, l):
    label = json.loads("".join(open(f"{l}").readlines()))
    img = cv2.imread(f"{s}", cv2.IMREAD_COLOR)

    for x in label:
        pts = x['points']
        x_min = min(p[0] for p in pts)
        x_max = max(p[0] for p in pts)
        y_min = min(p[1] for p in pts)
        y_max = max(p[1] for p in pts)

        word_img = img[y_min:y_max, x_min:x_max]

        try:
            pts = [[int(x[0]), int(x[1])] for x in pts]
        except:
            continue

        cropped_img = perspective_transform(img, pts)

        cv2.imshow("1", word_img)
        cv2.imshow("2", cropped_img)
        cv2.waitKey(0)

    return img


def order_points(pts):
    if isinstance(pts, list):
        pts = np.asarray(pts, dtype='float32')

    rect = np.zeros((4, 2), dtype='float32')
    s = pts.sum(axis=1)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def perspective_transform(img, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    return warped
