import sys
import torch

from cv2 import cv2

# yolo
sys.path.append("./yolov7/")

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device


class DetectWords():
    def __init__(self, weights, imgsz=640, conf_thres=0.5, iou_thres=0.5, device='', to_gray=False):
        self.weights = weights
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device
        self.to_gray = to_gray

        self.init()


    def init(self):
        self.device = select_device(self.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check img_size

        if self.half:
            self.model.half()  # to FP16

        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once

        self.old_img_w = self.old_img_h = self.imgsz
        self.old_img_b = 1


    def detect(self, source, min_h=10, min_w=10, return_locs=False):
        dataset = LoadImages(source, img_size=self.imgsz, stride=self.stride)

        # Run inference
        for _, img, im0s, _ in dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0

            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Warmup
            if self.device.type != 'cpu' and (self.old_img_b != img.shape[0] or self.old_img_h != img.shape[2] or self.old_img_w != img.shape[3]):
                self.old_img_b = img.shape[0]
                self.old_img_h = img.shape[2]
                self.old_img_w = img.shape[3]

                for _ in range(3):
                    self.model(img)[0]

            # Inference
            with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                pred = self.model(img)[0]

            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)

            # Process detections
            for det in pred:  # detections per image
                if len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()

                    comps = []
                    locs = [] 

                    for *x, _, _ in reversed(det):
                        if int(x[3] - x[1]) < min_h or int(x[2] - x[0]) < min_w:
                            continue

                        if return_locs:
                            locs.append((int(x[1]), int(x[3]), int(x[0]), int(x[2])))
                        else:
                            comps.append(im0s[int(x[1]):int(x[3]), int(x[0]):int(x[2])])

                    if self.to_gray:
                        comps = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in comps]

                    if return_locs:
                        return locs

                    return comps
