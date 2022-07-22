import torch
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, '/home/robesafe/Javier/3D-detection-system-lidar-camera/ultralytics_yolov5_master')
from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.augmentations import letterbox
from utils.torch_utils import select_device

class Yolo_Detector:
    
    def __init__(self, weigths, configuration, device=0):
        self.weigths = weigths
        self.configuration = configuration
        self.device = device

    @torch.no_grad()
    def build_model(self):
        device = select_device(self.device)
        self.model = DetectMultiBackend(self.weigths, device=device, dnn=False, data=self.configuration, fp16=False)
        self.stride, names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size((640, 640), s=self.stride)
        self.model.warmup(imgsz=(1, 3, *self.imgsz))  # warmup

    @torch.no_grad()
    def detect_yolo(self, img, conf_thres, iou_thres=0.45, max_det=10000):
        im = letterbox(img, self.imgsz, stride=self.stride, auto=self.pt)[0]
        im = np.rollaxis(im,1,0)
        im = torch.from_numpy(im).to(self.device).float()
        im /= 255  # 0 - 255 to 0.0 - 1.0
        im = im.T
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        pred = self.model(im)
        pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, max_det=max_det)
        
        for det in pred:
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], img.shape).round()

        return pred
