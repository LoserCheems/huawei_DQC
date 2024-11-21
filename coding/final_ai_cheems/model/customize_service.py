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


import os
import glob
import shutil
from detect_ma import Yolov5
from model_service.pytorch_model_service import PTServingBaseService

class yolov5_detection(PTServingBaseService):
    def __init__(self, model_name, model_path):
        print('model_name:',model_name)
        print('model_path:',model_path)
                
        self.model = Yolov5(model_path, device='cpu')
        
        self.capture = "test.png"

    def _preprocess(self, data):
        # preprocessed_data = {}
        for _, v in data.items():
            for _, file_content in v.items():
                with open(self.capture, 'wb') as f:
                    file_content_bytes = file_content.read()
                    f.write(file_content_bytes)
        return "ok"
    
    def _inference(self, data):
        pred_result = self.model.inference(self.capture)
        return pred_result

    def _postprocess(self, data):
        result = {}
        detection_classes = []
        detection_boxes = []
        detection_scores = []
        
        for pred in data:
            classes, _, x1, y1, x2, y2, conf = pred
            detection_classes.append(classes)
            boxes = [y1,x1,y2,x2]
            detection_boxes.append(boxes)
            detection_scores.append(conf)
                
        result['detection_classes'] = detection_classes
        result['detection_boxes'] = detection_boxes
        result['detection_scores'] = detection_scores
            
        print('result:',result)    
        return result