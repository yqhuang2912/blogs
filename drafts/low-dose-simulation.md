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

在医学成像中，低剂量CT（Computed Tomography）扫描是一种常用的技术，旨在减少患者接受的辐射剂量。然而，降低辐射剂量通常会导致图像质量下降，主要表现为噪声增加。为了研究和改进低剂量CT图像的处理方法，我们需要模拟低剂量CT图像中的噪声特性。从上一篇文章《[CT投影和泊松噪声的关系](./posts/ct-proj-and-poisson.html)》中，我们知道，在理想的、无噪声的世界里，X 射线穿过物体遵循 Beer-Lambert 定律:
$$N = N_0 \cdot e^{-\mu l} = N_0 \cdot e^{-p_{\text{ICT}}} \tag{1}$$
其中，$N_0$是入射到物体上的 X 射线光子数，$N$是穿过物体后到达探测器的 X 射线光子数，$\mu$是物体的线性衰减系数，$l$是射线穿过物体的路径长度，$p_{\text{ICT}}$是**理想线积分值**(True Line Integral)，也即CT的投影值，这是我们想要测量的物理量，则:
$$p_{\text{ICT}} = -\ln\left(\frac{N}{N_0}\right) \tag{2}$$

在现实世界中，光子的发射和探测是一个随机过程。即$N$不再是一个确定值，而是一个服从泊松分布的随机变量:
$$N \sim \text{Poisson}(N_0 \cdot e^{-p_{\text{ICT}}}) \tag{3}$$

在低剂量CT扫描中，入射光子的强度会降低，假设降低的比例为$\alpha$（$0 < \alpha < 1$），即$N_0^{'} = \alpha N_0$，那么低剂量CT下探测器接收到的光子数量可以表示为：
$$N_{\text{LDCT}} \sim \text{Poisson}(\alpha N_0 \cdot e^{-p_{\text{ICT}}}) \tag{4}$$
<!-- more -->

那么，我们只要从式(4)中采样，就可以得到模拟的低剂量CT系统中探测器接收到的光子数量$N_{\text{LDCT}}$。然后计算出对应的低剂量CT投影值$p_{\text{LDCT}}$：
$$p_{\text{LDCT}} = -\ln\left(\frac{N_{\text{LDCT}}}{N_0^{'}}\right) = -\ln\left(\frac{N_{\text{LDCT}}}{\alpha N_0}\right) \tag{5}$$

代码示例（Python）：

```python
import torch

def poisson_noise_simulation(p_ict, N0, alpha=1.0):
    """
    p_ict: full-dose sinogram (line integral)
    N0:    full-dose 入射光子数
    alpha: 剂量缩放因子 (0<alpha<=1)
    """
    N0_ldct = N0 * alpha                  # 低剂量下的 N0
    # 1) full-dose 透过率 -> 低剂量 photon 期望
    lam = N0_ldct * torch.exp(-p_ict)     # λ = N0_ldct * exp(-p)
    # 2) Poisson 噪声
    I_ldct = torch.poisson(lam)
    # 3) 反算回 line integral
    p_ldct = -torch.log(I_ldct / N0_ldct)
    return p_ldct
```
下面展示一个低剂量CT噪声模拟的示例，假设入射光子数$N_0=1e5$，剂量缩放因子$\alpha=0.25$（即25%剂量）的情况下的模拟结果：
<figure>
  <img src="./images/ldct_simulation.png" alt="低剂量CT噪声模拟示例", width=400>
  <figcaption>25%剂量CT噪声模拟示例</figcaption>
</figure>

## 投影域高斯噪声仿真
然而CT系统记录的不是$N$，而是通过对数变换得到的投影数据$p_{\text{measured}}$：
$$p_{\text{measured}} = -\ln\left(\frac{N}{N_0}\right) \tag{6}$$
由于$N$是随机的，$p_{\text{measured}}$ 也是一个随机变量。我们的任务就是精确地刻画$p_{\text{measured}}$的统计特性（均值和方差）。Whiting et al.(2006)在其论文的Section IV. Discussion中明确指出：
> "As exposure increases, the pdf will approach a Gaussian distribution characterized by a mean $\mu$ and variance $\sigma^2$, with negligible higher moments, as expected from the central limit theorem."

也就是说，当入射光子数$N_0$较大时，其概率密度函数会趋近于一个高斯分布，其高阶矩（如偏度、峰度）可以忽略不计，这是中心极限定理的预期结果。而且论文中还给出了结论：**当检测到的光子数$N > 20$时，泊松分布与高斯分布的差异已经非常小，高斯近似是足够精确的。**

