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
import torch_npu
from torch_npu.contrib import transfer_to_npu

# 加载模型
model = YOLO("yolov8x.pt", task="detect")
print(model)

# 训练模型
if __name__ == "__main__":
    results = model.train(

        data = "pcb_dataset.yaml", 
        imgsz = 2560,
        epochs = 16, 
        patience = 8,
        batch = 1,
        device = [0], 
        cos_lr = True,
        lr0 = 1e-3,
        lrf = 1e-3,
        weight_decay = 5e-4,
        warmup_epochs = 1,
        close_mosaic = 2,
        dropout = 0.1,
        plots = True,
        amp = False,
        workers = 1,
        label_smoothing = 0.0,
        
        augment = True,
        auto_augment = "augmix",
        hsv_h = 0.015,
        hsv_s = 0.7,
        hsv_v = 0.4,
        degrees = 0.0,
        translate = 0.1,
        scale = 0.9,
        shear = 0.0,
        perspective = 0.0001,
        flipud = 0.4,
        fliplr = 0.4,
        bgr = 0.01,
        mosaic = 1.0,
        mixup = 0.15,
        copy_paste = 0.3,
        erasing = 0.4,
        crop_fraction = 1.0,
    )
