<!-- coding=utf-8
Copyright 2024 Jingze Shi and Bingheng Wu.    All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License. -->

## Data Abstract Augmentation

常见的数据增强方法包括:

| 方法 | 类型 | 默认值 | 范围 | 说明 |
| --- | --- | --- | --- | --- |
| hsv_h | float | 0.2 | 0.0 - 1.0 | 通过色轮的一部分来调整图像的色调, 从而引入色彩的可变性. 帮助模型在不同的光照条件下通用化. |
| hsv_s | float | 0.6 | 0.0 - 1.0 | 改变图像饱和度的一部分, 影响色彩的强度. 可用于模拟不同的环境条件. |
| hsv_v | float | 0.4 | 0.0 - 1.0 | 将图像的亮度修改一部分, 帮助模型在不同的光照条件下表现良好. |
| degrees | float | 15.0 | -180 - +180 | 在指定的度数范围内随机旋转图像, 提高模型识别不同方向物体的能力. |
| translate | float | 0.1 | 0.0 - 1.0 | 以图像大小的一小部分水平和垂直平移图像, 帮助学习检测部分可见的物体. |
| scale | float | 0.5 | >= 0.0 | 通过增益因子缩放图像, 模拟物体与摄像机的不同距离. |
| shear | float | 15.0 | -180 - +180 | 按指定角度剪切图像, 模拟从不同角度观察物体的效果. |
| perspective | float | 0.0001 | 0.0 - 0.001 | 对图像进行随机透视变换, 增强模型理解三维空间中物体的能力. |
| flipud | float | 0.5 | 0.0 - 1.0 | 以指定的概率将图像翻转过来, 在不影响物体特征的情况下增加数据的可变性. |
| fliplr | float | 0.5 | 0.0 - 1.0 | 以指定的概率将图像从左到右翻转, 这对学习对称物体和增加数据集多样性非常有用 |
| bgr | float | 0.01 | 0.0 - 1.0 | 以指定的概率将图像通道从 RGB 翻转到 BGR, 用于提高对错误通道排序的稳健性. |
| mosaic | float | 1.0 | 0.0 - 1.0 | 将四幅训练图像合成一幅, 模拟不同的场景构成和物体互动, 对复杂场景的理解非常有效. |
| mixup | float | 0.5 | 0.0 - 1.0 | 混合两幅图像及其标签, 创建合成图像. 通过引入标签噪声和视觉变化, 增强模型的泛化能力. |
| copy_paste | float | 0.2 | 0.0 - 1.0 | 从一幅图像中复制物体并粘贴到另一幅图像上, 用于增加物体实例和学习物体遮挡. |
| erasing | float | 0.4 | 0.0 - 1.0 | 在分类训练过程中随机擦除部分图像, 鼓励模型将识别重点放在不明显的特征上. |
| crop_fraction | float | 1.0 | 0.0 - 1.0 | 将分类图像裁剪为其大小的一小部分, 以突出中心特征并适应对象比例, 减少背景干扰. |

我们定义增强方法的集合为 $A$, 增强方法的概率集合为 $p$, $n$ 为样本的数量, $m$ 为增强方法的数量, 那么我们的抽象增强方法可以表示为:

$$
Abs_{Aug}(example) = \sum_{i=1}^{n} \sum_{j=1}^{m} p_{j} \circ A_{j}(example_{i})
$$

