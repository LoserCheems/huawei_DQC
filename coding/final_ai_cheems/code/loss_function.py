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



# 非均匀样本动态加权损失函数
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

# 引入 loss 父类
from torch.nn.modules.loss import  _WeightedLoss


class CheemsLoss(_WeightedLoss):
    __doc__ = r"""
    动态关注加权损失函数
    Args:
        weighted_terms: 加权索引
        weighted_type: 加权类型
        weighted_rate: 加权率
        weighted_base: 基础权重
        weighted_freq: 曲线频率
    Inputs:
        logits: [batch_size, label_size], 预测标签
        labels: [batch_size], 真实标签
    Returns:
        loss: [1], 损失值
    """
    def __init__(
        self,
        weighted_terms: List[int] = None,
        weighted_type: str = 'sin',
        weighted_rate: float = 1.0,
        weighted_base: float = 1,
        weighted_freq: float = 1.0,
    ) -> None:
        super().__init__()
        self.weighted_terms = weighted_terms
        self.weighted_type = weighted_type
        self.weighted_rate = weighted_rate
        self.weighted_base = weighted_base
        self.weighted_freq = weighted_freq

    def weight(
        self,
        logits: torch.Tensor,
        weighted_terms: List[int] = None,
        weighted_type: str = 'sin',
        weighted_rate: float = 1.0,
        weighted_base: float = 1,
        weighted_freq: float = 1.0,
    ) -> torch.Tensor:
        """
        Inputs:
            logits: [batch_size, label_size], 预测分数
            weighted_terms: 加权索引
            weighted_type: 加权类型
            weighted_rate: 加权率
            weighted_base: 基础权重
            weighted_freq: 曲线频率
        Returns:
            weight: [vocab_size], 权重
        """
        
        # 获取标签大小
        label_size = logits.size(-1)
        # 获取设备
        device = logits.device
        x = logits / torch.max(logits) # 归一化
        
        # 如果加权索引不为空
        if weighted_terms is not None:
            # 初始化权重 [batch_size, label_size]
            weight = torch.zeros(label_size, dtype=torch.bool, device=device)
            weight[weighted_terms] = True
            # 如果加权类型为正弦波
            if weighted_type == 'sin':
                # 如果预测词概率为需要加权类别
                weights = torch.where(
                    weight == True,

                    weighted_rate * torch.sum(torch.sin(torch.pi / weighted_freq * x), dim=0), # 则以正弦波的形式增大 [label_size]
                    
                    torch.sum(x, dim=0) # 否则保证其在合理范围内
                )
            if weighted_type == 'cos':
                # 如果预测词概率为需要加权类别
                weights = torch.where(
                    weight == True, 

                    weighted_rate * torch.sum(torch.cos(torch.pi / weighted_freq * x), dim=0), # 则以余弦波的形式增大 [label_size]

                    torch.sum(x, dim=0) # 否则保证其在合理范围内
                )
        else:
            weights = torch.sum(x, dim=0) # 否则保证其在合理范围内

        # 与基础权重相加
        weights = weighted_base + F.softmax(weights, dim=-1)
       
        # nan 或 inf 处理
        weights = torch.where(
            torch.isnan(weights) | torch.isinf(weights),
            torch.tensor(1.0, device=device),
            weights
        )
        return weights

    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss_fct: nn.CrossEntropyLoss = None,
    ) -> torch.Tensor:
        # 计算权重
        weights = self.weight(
            logits,
            self.weighted_terms, 
            self.weighted_type, 
            self.weighted_rate,
            self.weighted_base,
            self.weighted_freq
        ).detach()
        # 计算损失

        # 交叉熵
        if loss_fct.__class__.__name__ == 'CrossEntropyLoss':
            loss = F.cross_entropy(logits, labels, weight=weights)

        # 二元交叉熵
        if loss_fct.__class__.__name__ == 'BCEWithLogitsLoss':
            loss = F.binary_cross_entropy_with_logits(logits, labels, weight=weights)

        return loss
