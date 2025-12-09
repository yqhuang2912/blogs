---
title: "为什么生成模型要使用MSE？"
createdAt: 2025-12-09
categories:
  - 人工智能
  - 数学研究
tags:
  - 图像生成
  - 均方误差
---

在生成模型中，尤其是图像生成任务中，均方误差（Mean Squared Error, MSE）是一种常用的损失函数。那么，为什么生成模型要使用MSE呢？本文将从概率建模的角度来解释这个问题。

## 定义生成任务
假设我们有一个生成任务，比如图像超分辨率或者CT图像重建。输入观测数据$z$，这里的$z$可以是低分辨率图像，也可以是CT的sinogram数据，在真实世界中客观存在一个映射关系$f^{\ast}$，使得我们可以从$z$生成高质量的图像$x$。但是，这个真实的映射关系$f^{\ast}$通常是未知的，我们只能通过数据来学习一个近似的映射函数$f_{\theta}$，使得$f_{\theta}(z)$产生的数据$\hat{x}$尽可能接近真实的映射关系$f^{\ast}(z)$产生的真实图像$x$。

在生成模型的视角下，生成不仅仅是预测一个具体的图像，而是希望学习一个条件概率分布$p(x|z; \theta)$，表示在给定观测数据$z$的情况下，真实图像$x$的分布。

<!-- more -->

我们现在有的是什么？我们可以从现实中收集到一组训练数据集$\mathcal{D} = \{(x_n, z_n)\}_{n=1}^{N}$，这些数据是从真实分布$p_{\text{data}}(x, z): z \to f^{\ast}(z)=x$中采样得到的。我们的终极目标是找到一个好的参数$\theta$，使得模型在观测数据$z$作为条件下生成的数据$\hat{x}$就好像是从真实分布$p_{\text{data}}(x,z)$中采样得到的一样。根据之前的《[概率和似然](probability-and-likelihood.html)》一文，我们可以通过最大化似然估计来实现这个目标：
$$
\theta^{\ast} = \arg\max_{\theta} \prod_{i=1}^{N} p(x_i | z_i; \theta) \tag{1}
$$
为了简化计算，我们通常取<span style='color: red'>负对数似然</span>，将乘法转换为加法，将最大化问题转换为最小化问题：
$$
\theta^{\ast} = \arg\min_{\theta} - \sum_{i=1}^{N} \log p(x_i | z_i; \theta) \tag{2}
$$
这就是生成问题的最一般化的目标函数的形式。接下来我们面临一个关键问题：**条件概率分布$p(x|z)$到底长啥样？**

## 为什么选择高斯分布？
我们换个思路：我们想要生成的图像$\hat{x}$应该和真实图像$x$尽可能接近，但是，什么叫“尽可能接近”呢？就是这两者之间的差异要尽可能小。我们使用$\epsilon$来表示这个差异，则有：
$$x = \hat{x} + \epsilon = f_{\theta}(z) + \epsilon \tag{3}$$

这意味着，$p(x|z)$的形状完全取决于$\epsilon$的分布的形状。

在实际应用中，我们通常假设$\epsilon$服从均值为0，方差为$\sigma^2 I$的高斯分布，这是为什么呢？这里有两个“第一性原理”的支撑。

### 中心极限定理--物理视角
在CT图像重建或者自然图像生成任务中，生成的图像$\hat{x}$与真实图像$x$之间的差异$\epsilon$往往不是由单一因素造成的，而是无数微小干扰的叠加：
1. 由于量子随机性带来的光子散粒噪声；
2. 由于布朗运动带来的电子热噪声；
3. 由于传感器缺陷带来的系统误差；
4. 由于电路量化误差带来的数值误差；
5. 由于数值计算近似带来的舍入误差；
6. ...

**中心极限定理**告诉我们：
> 只要存在大量相互独立的随机变量，无论它们原本服从什么分布（均匀，二项，泊松等），它们的和都会趋近于高斯分布：
> $$\epsilon_{\text{total}} = \sum_{i}^n \epsilon_i \xrightarrow{n \to \infty} \mathcal{N}(0, \sigma^2 I) \tag{4}$$
因此假设误差$\epsilon$服从高斯分布，是对复杂物理世界最合理的近似。

### 最大熵原理--信息论视角
从信息论的角度来看，假设我们对误差$\epsilon$的分布一无所知，只知道两条基本的物理限制：
1. **无偏性**：误差的均值为0，即$\mathbb{E}[\epsilon] = 0$；
2. **有限能量**：根据物理学的能量守恒定律，误差不可能无限大，因此误差的方差是有限的，我们不妨设为$\sigma^2$，即$\mathbb{E}[\epsilon^2] = \sigma^2$。

**最大熵原理**告诉我们：
> 在所有满足上述约束条件的分布中，熵最大的分布是最合理的选择，因为它对未知信息的假设最少，最不偏袒任何特定的分布形式。

这是一个泛函极值问题：
$$
\begin{aligned}
    &\max_{p}H(p) = -\int p(\epsilon) \log p(\epsilon) d\epsilon \\
    &\text{s.t.} \quad \int p(\epsilon) d\epsilon = 1, \ \int \epsilon^2 p(\epsilon) d\epsilon = \sigma^2
\end{aligned} \tag{5}
$$
通过引入拉格朗日乘子$\lambda_0$和$\lambda_1$，我们可以构造拉格朗日函数：
$$
\begin{aligned}
    \mathcal{L}(p, \lambda_0, \lambda_1) &= -\int p(\epsilon) \log p(\epsilon) d\epsilon \\
    &+ \lambda_0 \left( \int p(\epsilon) d\epsilon - 1 \right) \\
    &+ \lambda_1 \left( \int \epsilon^2 p(\epsilon) d\epsilon - \sigma^2 \right)
\end{aligned}
\tag{6}
$$
通过对$p(\epsilon)$求变分，并令其为零，我们可以得到：
$$\frac{\delta \mathcal{L}}{\delta p(\epsilon)} = -\log p(\epsilon) - 1 + \lambda_0 + \lambda_1 \epsilon^2 = 0 \tag{7}$$
解出$p(\epsilon)$，我们得到：
$$p(\epsilon) = \exp(\lambda_0 - 1) \exp(-\lambda_1 \epsilon^2) \tag{8}$$
把这个结果代入第一个约束条件中，有：
$$
\begin{aligned}
    \int p(\epsilon) d\epsilon &= \int \exp(\lambda_0 - 1) \exp(-\lambda_1 \epsilon^2) d\epsilon\\
    & = \exp(\lambda_0 - 1) \int \exp(-\lambda_1 \epsilon^2) d\epsilon \\
    & = \exp(\lambda_0 - 1) \sqrt{\frac{\pi}{\lambda_1}} = 1 \\
    &\Rightarrow \exp(\lambda_0 - 1) = \sqrt{\frac{\lambda_1}{\pi}}
\end{aligned} \tag{9}
$$
带入第二个约束条件中，有：
$$
\begin{aligned}
    \int \epsilon^2 p(\epsilon) d\epsilon &= \int \epsilon^2 \exp(\lambda_0 - 1) \exp(-\lambda_1 \epsilon^2) d\epsilon \\
    & = \exp(\lambda_0 - 1) \int \epsilon^2 \exp(-\lambda_1 \epsilon^2) d\epsilon \\
    & = \exp(\lambda_0 - 1) \frac{1}{2\lambda_1^{3/2}} \sqrt{\pi} = \sigma^2 \\
    &\Rightarrow \exp(\lambda_0 - 1) = 2 \lambda_1^{3/2} \frac{\sigma^2}{\sqrt{\pi}}
\end{aligned} \tag{10}
$$
联立上面(9)和(10)两个方程，解出$\lambda_1$，有：
$$\lambda_1 = \frac{1}{2\sigma^2} \tag{11}$$

将$\lambda_1$代入(9)式中，解出$\lambda_0$，有：
$$\exp(\lambda_0 - 1) = \frac{1}{\sqrt{2\pi \sigma^2}} \tag{12}$$
将$\lambda_0$和$\lambda_1$代入(8)式中，最终得到：
$$p(\epsilon) = \frac{1}{\sqrt{2\pi \sigma^2}} \exp\left(-\frac{\epsilon^2}{2\sigma^2}\right) \tag{13}$$
这正是均值为$0$，方差为$\sigma^2$的高斯分布。

因此，从**中心极限定理**和**最大熵原理**两个角度来看，假设误差是高斯的，是因为这是在已知方差限制下，**最诚实、偏差最少**的假设。

## 从高斯分布到MSE
既然基于**中心极限定理**和**最大熵原理**确定了误差$\epsilon$服从高斯分布：$\epsilon \sim \mathcal{N}(0, \sigma^2)$，那么条件概率分布$p(x|z)$也就确定下来了：
$$p(x|z; \theta) = \mathcal{N}(x; f_{\theta}(z), \sigma^2 I) \tag{14}$$
将(14)式中的结果代入(13)式中，我们可以得到：
$$p(x|z; \theta) = \frac{1}{\sqrt{2\pi \sigma^2}} \exp\left(-\frac{(x - f_{\theta}(z))^2}{2\sigma^2}\right) \tag{15}$$
将(15)式代入最大似然估计的负对数形式(2)中，有：
$$
\begin{aligned}
    \theta^{\ast} &= \arg\min_{\theta} - \sum_{i=1}^{N} \log p(x_i | z_i; \theta) \\
    &= \arg\min_{\theta} - \sum_{i=1}^{N} \log \left[ \frac{1}{\sqrt{2\pi \sigma^2}} \exp\left(-\frac{(x_i - f_{\theta}(z_i))^2}{2\sigma^2}\right) \right] \\
    &= \arg\min_{\theta} - \sum_{i=1}^{N} \left[\log\left(\frac{1}{\sqrt{2\pi \sigma^2}}\right) - \frac{(x_i - f_{\theta}(z_i))^2}{2\sigma^2}\right] \\
\end{aligned} \tag{16}
$$
注意到$\log\left(\frac{1}{\sqrt{2\pi \sigma^2}}\right)$是一个常数项，与参数$\theta$无关，因此在优化过程中可以忽略。最终，我们得到：
$$
\theta^{\ast} = \arg\min_{\theta} \sum_{i=1}^{N} (x_i - f_{\theta}(z_i))^2 \tag{17}
$$
这正是均方误差（MSE）损失函数的形式。