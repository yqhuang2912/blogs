---
title: "低剂量CT图像的噪声模拟"
createdAt: 2025-11-20
categories:
  - 医学图像
  - 动手实践
tags:
  - CT成像
  - 泊松噪声
---

在医学成像中，低剂量CT（Computed Tomography）扫描是一种常用的技术，旨在减少患者接受的辐射剂量。然而，降低辐射剂量通常会导致图像质量下降，主要表现为噪声增加。为了研究和改进低剂量CT图像的处理方法，我们需要模拟低剂量CT图像中的噪声特性。从上一篇文章《[CT投影和泊松噪声的关系](./posts/ct-proj-and-poisson.html)》中，我们知道，在正常剂量下，探测器接收到的光子数量服从泊松分布：
$$I \sim \text{Poisson}(I_0 \cdot e^{-p_{\text{ICT}}}) \tag{1}$$
其中，$I_0$是入射光子的强度，$p_{\text{ICT}}$是理想CT图像的投影值。

而在低剂量CT扫描中，入射光子的强度会降低，假设降低的比例为$\alpha$（$0 < \alpha < 1$），即$I_0^{'} = \alpha I_0$，那么低剂量CT下探测器接收到的光子数量可以表示为：
$$I_{\text{LDCT}} \sim \text{Poisson}(\alpha I_0 \cdot e^{-p_{\text{ICT}}}) \tag{2}$$
<!-- more -->

那么，我们只要从式(2)中采样，就可以得到模拟的低剂量CT系统中探测器接收到的光子数量$I_{\text{LDCT}}$。然后计算出对应的低剂量CT投影值$p_{\text{LDCT}}$：
$$p_{\text{LDCT}} = -\ln\left(\frac{I_{\text{LDCT}}}{I_0^{'}}\right) = -\ln\left(\frac{I_{\text{LDCT}}}{\alpha I_0}\right) \tag{3}$$

代码示例（Python）：

```python
import numpy as np

def simulate_low_dose_sino(p_ict, I0, alpha):
    # 计算低剂量下的入射光子强度
    I0_ldct = alpha * I0
    # 计算泊松分布的参数
    lambda_ldct = I0_ldct * np.exp(-p_ict)
    # 从泊松分布中采样得到探测器接收到的光子数量
    I_ldct = np.random.poisson(lambda_ldct)
    # 计算低剂量CT投影值
    p_ldct = -np.log(I_ldct / (I0_ldct + 1e-10))
    return p_ldct