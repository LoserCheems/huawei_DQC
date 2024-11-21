# coding=utf-8
# Copyright 2024 Jingze Shi and Bingheng Wu.    All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.



from ultralytics import YOLO
import os
import sys
from pathlib import Path
import torch


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.datasets import LoadImages
from utils.general import check_img_size, scale_coords, increment_path, set_logging
from utils.torch_utils import select_device


num2cls = {
    0: "Mouse_bite",
    1: "Open_circuit",
    2: "Short",
    3: "Spur",
    4: "Spurious_copper"
}

class Yolov5():
    def __init__(self, weights_path, device='cpu'):
        self.model = YOLO(weights_path, task="detect")
        self.imgsz = [2048, 3072]  # inference size (pixels)
        self.conf_thres = 0.25
        self.iou_thres = 0.5
        self.max_det = 20
        self.augment = False
        self.device = device
        self.project = ROOT / 'runs/detect'

        set_logging()
        self.device = select_device(device)
        
        self.stride = 32
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size

    @torch.no_grad()
    def inference(self, source):

        dataset = LoadImages(source, img_size=self.imgsz, stride=self.stride, auto=True)

        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.float()
            img /= 255.0
            
            if len(img.shape) == 3:
                img = img[None]
            
            pred = self.model.predict(img, conf=self.conf_thres, iou=self.iou_thres, max_det=self.max_det, augment=self.augment, device=self.device)[0]

            detection_classes_idx = pred.boxes.cls.tolist()
            if detection_classes_idx != []:
                detection_classes_name = [num2cls[i] for i in detection_classes_idx]
            detection_boxes = pred.boxes.xyxy.tolist()
            # 缩放detection_boxes到原图大小
            if detection_boxes != []:
                detection_boxes = torch.tensor(detection_boxes)
                detection_boxes = scale_coords(img.shape[2:], detection_boxes, im0s.shape).round().tolist()

            detection_scores = pred.boxes.conf.tolist()

            pred_result = []
            for i in range(len(detection_classes_idx)):
                pred_result.append([detection_classes_name[i], detection_classes_idx[i], detection_boxes[i][0], detection_boxes[i][1], detection_boxes[i][2], detection_boxes[i][3], detection_scores[i]])

            print(pred_result)
            return pred_result