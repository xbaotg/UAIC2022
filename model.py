import pickle
import sys
import traceback

import cv2
import numpy as np
import torch
from tqdm import tqdm

# YOLOv7
sys.path.append("YOLOv7")
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device

# PaddleOCR
sys.path.append("PaddleOCR")
import tools.infer.utility as utility
from ppocr.postprocess import build_post_process
from ppocr.utils.logging import get_logger
from ppocr.utils.utility import get_image_file_list, check_and_read


# {{{ Recode from YOLOv7 detect.py
class DetectWords():
    def __init__(self, weights, imgsz=640, conf_thres=0.5, iou_thres=0.5, device=''):
        self.weights = weights
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device

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
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(
                next(self.model.parameters())))  # run once

        self.old_img_w = self.old_img_h = self.imgsz
        self.old_img_b = 1

    def detect(self, source, min_h=10, min_w=10) -> tuple:
        dataset = LoadImages(source, img_size=self.imgsz, stride=self.stride)

        # Run inference
        for _, img, im0s, _ in dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0

            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Warmup
            if self.device.type != 'cpu' and (
                    self.old_img_b != img.shape[0] or self.old_img_h != img.shape[2] or self.old_img_w != img.shape[3]):
                self.old_img_b = img.shape[0]
                self.old_img_h = img.shape[2]
                self.old_img_w = img.shape[3]

                for _ in range(3):
                    self.model(img)[0]

            # Inference
            with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
                pred = self.model(img)[0]

            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)

            # Process detections
            for det in pred:  # detections per image
                if len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()

                    locs = []

                    for *x, _, _ in reversed(det):
                        if int(x[3] - x[1]) < min_h or int(x[2] - x[0]) < min_w:
                            continue
                        locs.append((int(x[1]), int(x[3]), int(x[0]), int(x[2])))

                    return (locs, im0s)

        return (None, None)


# }}}
# {{{ Recode PaddleOCR 
class TextRecognizer(object):
    def __init__(self, dict_path="./configs/dict.txt"):
        args = pickle.load(open("./configs/config_ocr.p", "rb"))
        self.rec_image_shape = [int(v) for v in args.rec_image_shape.split(",")]
        self.rec_batch_num = args.rec_batch_num
        self.rec_algorithm = args.rec_algorithm

        postprocess_params = {
            'name': 'SRNLabelDecode',
            "character_dict_path": dict_path,
            "use_space_char": False
        }

        self.logger = get_logger()
        self.postprocess_op = build_post_process(postprocess_params)
        self.predictor, self.input_tensor, self.output_tensors, self.config = \
            utility.create_predictor(args, 'rec', self.logger)

    def resize_norm_img_srn(self, img, image_shape):
        _, imgH, imgW = image_shape

        img_black = np.zeros((imgH, imgW))
        im_hei = img.shape[0]
        im_wid = img.shape[1]

        if im_wid <= im_hei * 1:
            img_new = cv2.resize(img, (imgH * 1, imgH))
        elif im_wid <= im_hei * 2:
            img_new = cv2.resize(img, (imgH * 2, imgH))
        elif im_wid <= im_hei * 3:
            img_new = cv2.resize(img, (imgH * 3, imgH))
        else:
            img_new = cv2.resize(img, (imgW, imgH))

        img_np = np.asarray(img_new)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        img_black[:, 0:img_np.shape[1]] = img_np
        img_black = img_black[:, :, np.newaxis]

        row, col, c = img_black.shape
        c = 1

        return np.reshape(img_black, (c, row, col)).astype(np.float32)

    def srn_other_inputs(self, image_shape, num_heads, max_text_length):
        _, imgH, imgW = image_shape
        feature_dim = int((imgH / 8) * (imgW / 8))

        encoder_word_pos = np.array(range(0, feature_dim)).reshape(
            (feature_dim, 1)).astype('int64')
        gsrm_word_pos = np.array(range(0, max_text_length)).reshape(
            (max_text_length, 1)).astype('int64')

        gsrm_attn_bias_data = np.ones((1, max_text_length, max_text_length))
        gsrm_slf_attn_bias1 = np.triu(gsrm_attn_bias_data, 1).reshape(
            [-1, 1, max_text_length, max_text_length])
        gsrm_slf_attn_bias1 = np.tile(
            gsrm_slf_attn_bias1,
            [1, num_heads, 1, 1]).astype('float32') * [-1e9]

        gsrm_slf_attn_bias2 = np.tril(gsrm_attn_bias_data, -1).reshape(
            [-1, 1, max_text_length, max_text_length])
        gsrm_slf_attn_bias2 = np.tile(
            gsrm_slf_attn_bias2,
            [1, num_heads, 1, 1]).astype('float32') * [-1e9]

        encoder_word_pos = encoder_word_pos[np.newaxis, :]
        gsrm_word_pos = gsrm_word_pos[np.newaxis, :]

        return [
            encoder_word_pos, gsrm_word_pos, gsrm_slf_attn_bias1,
            gsrm_slf_attn_bias2
        ]

    def process_image_srn(self, img, image_shape, num_heads, max_text_length):
        norm_img = self.resize_norm_img_srn(img, image_shape)
        norm_img = norm_img[np.newaxis, :]

        [encoder_word_pos, gsrm_word_pos, gsrm_slf_attn_bias1, gsrm_slf_attn_bias2] = \
            self.srn_other_inputs(image_shape, num_heads, max_text_length)

        gsrm_slf_attn_bias1 = gsrm_slf_attn_bias1.astype(np.float32)
        gsrm_slf_attn_bias2 = gsrm_slf_attn_bias2.astype(np.float32)
        encoder_word_pos = encoder_word_pos.astype(np.int64)
        gsrm_word_pos = gsrm_word_pos.astype(np.int64)

        return (norm_img, encoder_word_pos, gsrm_word_pos, gsrm_slf_attn_bias1,
                gsrm_slf_attn_bias2)

    def recognize(self, image_file_list):
        image_file_list = get_image_file_list(image_file_list)
        img_list = []

        for image_file in image_file_list:
            img, flag, _ = check_and_read(image_file)
            if not flag:
                img = cv2.imread(image_file)
            if img is None:
                self.logger.info("error in loading image:{}".format(image_file))
                continue

            img_list.append(img)

        try:
            img_num = len(img_list)
            width_list = []

            for img in img_list:
                width_list.append(img.shape[1] / float(img.shape[0]))

            indices = np.argsort(np.array(width_list))
            rec_res = [['', 0.0]] * img_num
            batch_num = self.rec_batch_num

            for beg_img_no in tqdm(range(0, img_num, batch_num)):
                end_img_no = min(img_num, beg_img_no + batch_num)
                norm_img_batch = []
                if self.rec_algorithm == "SRN":
                    encoder_word_pos_list = []
                    gsrm_word_pos_list = []
                    gsrm_slf_attn_bias1_list = []
                    gsrm_slf_attn_bias2_list = []

                _, imgH, imgW = self.rec_image_shape[:3]
                max_wh_ratio = imgW / imgH
                # max_wh_ratio = 0
                for ino in range(beg_img_no, end_img_no):
                    h, w = img_list[indices[ino]].shape[0:2]
                    wh_ratio = w * 1.0 / h
                    max_wh_ratio = max(max_wh_ratio, wh_ratio)

                for ino in range(beg_img_no, end_img_no):
                    norm_img = self.process_image_srn(
                        img_list[indices[ino]], self.rec_image_shape, 8, 25)

                    encoder_word_pos_list.append(norm_img[1])
                    gsrm_word_pos_list.append(norm_img[2])
                    gsrm_slf_attn_bias1_list.append(norm_img[3])
                    gsrm_slf_attn_bias2_list.append(norm_img[4])
                    norm_img_batch.append(norm_img[0])

                norm_img_batch = np.concatenate(norm_img_batch)
                norm_img_batch = norm_img_batch.copy()

                if self.rec_algorithm == "SRN":
                    encoder_word_pos_list = np.concatenate(encoder_word_pos_list)
                    gsrm_word_pos_list = np.concatenate(gsrm_word_pos_list)
                    gsrm_slf_attn_bias1_list = np.concatenate(
                        gsrm_slf_attn_bias1_list)
                    gsrm_slf_attn_bias2_list = np.concatenate(
                        gsrm_slf_attn_bias2_list)

                    inputs = [
                        norm_img_batch,
                        encoder_word_pos_list,
                        gsrm_word_pos_list,
                        gsrm_slf_attn_bias1_list,
                        gsrm_slf_attn_bias2_list,
                    ]

                    input_names = self.predictor.get_input_names()
                    for i in range(len(input_names)):
                        input_tensor = self.predictor.get_input_handle(
                            input_names[i])
                        input_tensor.copy_from_cpu(inputs[i])

                    self.predictor.run()
                    outputs = []
                    for output_tensor in self.output_tensors:
                        output = output_tensor.copy_to_cpu()
                        outputs.append(output)

                    preds = {"predict": outputs[2]}

                rec_result = self.postprocess_op(preds)
                for rno in range(len(rec_result)):
                    rec_res[indices[beg_img_no + rno]] = rec_result[rno]

        except Exception as E:
            self.logger.info(traceback.format_exc())
            self.logger.info(E)
            exit()

        return image_file_list, rec_res
# }}}
