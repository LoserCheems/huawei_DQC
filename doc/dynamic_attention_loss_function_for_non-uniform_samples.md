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

# 非均匀样本动态关注损失函数

假如我们的工业样本类别分布有少量的不均衡, 复制粘贴的数据扩充方法会使得不均衡问题更加严重, 即使使用数据抽象增强, 也只是增加数据的多样性, 并不能改变原始的类别比例.

<img src="../assets/non-uniform_samples.jpg" width="50%">

为了解决这个问题, 我们提出了一种 `非均匀样本动态关注损失函数`, 使得模型在训练过程中动态调整不同类别的损失权重, 以此来解决样本不均衡问题.

$$
weight =
    \sum_{l=1}^{label} \cos (\frac{\pi}{freq} \times \frac{logits_{l}}{\max(logits_{l})}, dim=0)
$$

$$
L := \ell(logits, label) = \{l_1,\dots,l_N\}^\top 
$$

$$
l_n = - \sum_{l=1}^{label} weight_{l} \times \log \frac{\exp(logits_{n,l})}{\sum_{l'=1}^{label} \exp(logits_{n,l'})} \times label_{n}
$$

其中 $logits$ 是模型的输出分数, $label$ 是真实标签, $weight$ 是动态权重, $freq$ 是频率, $L$ 是损失函数, $l_n$ 是第 $n$ 个样本的损失函数, $label_{n}$ 是第 $n$ 个样本的真实标签, $logits_{n,l}$ 是第 $n$ 个样本的第 $l$ 个类别的预测分数.

推荐使用 $freq$ 为:
$$
freq = 1 + \frac{1}{\sqrt{\frac{1}{n}\sum_{i=1}^{n} (x_i - \bar{x})^2 + \epsilon}}
$$

其中 $n$ 是样本类别数量, $x_i$ 是样本类别数量的频率, $\bar{x}$ 是样本类别数量的平均值, $\epsilon$ 是一个很小的数来防止分母为零.

首先将预测分数转换为从0到1的概率分布, 使其与最大概率分布相除, 从而得到一个相对的概率分布.然后将其通过余弦函数, 从而得到一个动态的权重: 当预测分数越大时, 权重越小; 当预测分数越小时, 权重越大.

因为它是动态的, 所以它不仅可以解决样本不均衡的问题, 还可以在一定程度上解决模型过拟合的问题. 当然, 在模型最终收敛时, 我们可以将权重固定为一个常数, 以此来保持模型的稳定性.