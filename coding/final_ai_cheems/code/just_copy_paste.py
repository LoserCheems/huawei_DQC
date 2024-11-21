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




# 获取当前路径下的文件
import os
import shutil
import cv2
from PIL import Image
from tqdm import tqdm
from glob import glob


# 获取当前路径
PATH = os.path.abspath(os.path.dirname(__file__))
# 输出目标路径
TARGET_PATH = f"{PATH}/PCB"

# 样例集名称
ONE_SET = "PCB_preliminary"
TWO_SET = "PCB_rematch"


# 首先将样例数据集中的bmp格式图片转换为jpg格式图片
def convert_bmp_to_jpg(path):
    for img_path in tqdm(glob(path)):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img.save(img_path.replace(".bmp", ".jpg"))
        os.remove(img_path)

convert_bmp_to_jpg(f"./{ONE_SET}/*/*.bmp")
convert_bmp_to_jpg(f"./{TWO_SET}/*/*.bmp")

# 获取当前路径下的文件
def get_file_list(path):
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list


Mouse_bite_images_file_list = get_file_list(f"{PATH}/{ONE_SET}/Mouse_bite_Img/")
Mouse_bite_labels_file_list = get_file_list(f"{PATH}/{ONE_SET}/Mouse_bite_txt/")
Open_circuit_images_file_list = get_file_list(f"{PATH}/{ONE_SET}/Open_circuit_Img/")
Open_circuit_labels_file_list = get_file_list(f"{PATH}/{ONE_SET}/Open_circuit_txt/")
Short_images_file_list = get_file_list(f"{PATH}/{ONE_SET}/Short_Img/")
Short_labels_file_list = get_file_list(f"{PATH}/{ONE_SET}/Short_txt/")
Spur_images_file_list = get_file_list(f"{PATH}/{ONE_SET}/Spur_Img/")
Spur_labels_file_list = get_file_list(f"{PATH}/{ONE_SET}/Spur_txt/")
Spurious_copper_images_file_list = get_file_list(f"{PATH}/{ONE_SET}/Spurious_copper_Img/")
Spurious_copper_labels_file_list = get_file_list(f"{PATH}/{ONE_SET}/Spurious_copper_txt/")

hybrid_images_file_list = get_file_list(f"{PATH}/{TWO_SET}/images/")
hybrid_labels_file_list = get_file_list(f"{PATH}/{TWO_SET}/labels/")


# 复制文件
def copy_image_files(file_list, copy_num=100, target_folder=TARGET_PATH, sub_set='train'):
    if not os.path.exists(f"{target_folder}/{sub_set}/images"):
        os.makedirs(f"{target_folder}/{sub_set}/images")
    for file in file_list:
        file_name = os.path.basename(file)
        for i in range(0, 0+copy_num):
            # 复制Img文件到total_images文件夹
            new_file = os.path.join(f'{target_folder}/{sub_set}/images', file_name.replace('.jpg', f'_{i+1}.jpg'))
            shutil.copy(file, new_file)
            print(f'copy {file} to {new_file}')


def copy_labels_files(file_list, copy_num=100, target_folder=TARGET_PATH, sub_set='train'):
    if not os.path.exists(f"{target_folder}/{sub_set}/labels"):
        os.makedirs(f"{target_folder}/{sub_set}/labels")
    for file in file_list:
        file_name = os.path.basename(file)
        for i in range(0, 0+copy_num):
            new_file = os.path.join(f'{target_folder}/{sub_set}/labels', file_name.replace('.txt', f'_{i+1}.txt'))
            shutil.copy(file, new_file)
            print(f'copy {file} to {new_file}')


# 初赛数据200份
num = 200
copy_image_files(Mouse_bite_images_file_list, copy_num=num, sub_set='train')
copy_labels_files(Mouse_bite_labels_file_list, copy_num=num, sub_set='train')
copy_image_files(Open_circuit_images_file_list, copy_num=num, sub_set='train')
copy_labels_files(Open_circuit_labels_file_list, copy_num=num, sub_set='train')
copy_image_files(Short_images_file_list, copy_num=num, sub_set='train')
copy_labels_files(Short_labels_file_list, copy_num=num, sub_set='train')
copy_image_files(Spur_images_file_list, copy_num=num, sub_set='train')
copy_labels_files(Spur_labels_file_list, copy_num=num, sub_set='train')
copy_image_files(Spurious_copper_images_file_list, copy_num=num, sub_set='train')
copy_labels_files(Spurious_copper_labels_file_list, copy_num=num, sub_set='train')

# 复赛数据500份
num = 500
copy_image_files(hybrid_images_file_list, copy_num=num, sub_set='train')
copy_labels_files(hybrid_labels_file_list, copy_num=num, sub_set='train')


# 验证集数据
copy_image_files(hybrid_images_file_list, copy_num=1, sub_set='val')
copy_labels_files(hybrid_labels_file_list, copy_num=1, sub_set='val')
