---
title: "L1解决了MSE的什么问题？"
createdAt: 2025-12-12
categories:
  - 人工智能
  - 数学研究
tags:
  - 图像生成
  - Laplace分布
  - L1损失
---


在上一篇《[为什么生成模型要使用MSE？](generative-model-mse.html)》中，我们从概率建模的角度推导了一个结论：如果假设误差:$x = f_\theta(y) + \epsilon$中的噪声$\epsilon$服从高斯分布，那么最大化似然就等价于最小化$(x - f_\theta(y))^2$，也就是**MSE损失**。

MSE看起来非常自然，也有漂亮的数学和物理背景：中心极限定理 + 最大熵。但在实际的生成模型中，比如图像去噪、超分、CT重建、超分辨率等，很多人会发现：仅仅使用MSE会带来一些问题，比如图像发糊、细节被抹平、对少量异常样本极度敏感。

## MSE本质上逼近的是均值

我们先把条件$y$固定下来，只关注「在这个$y$下，$x$的真实分布是什么」。设真实数据来自$p_{\text{data}}(x|y)$，模型在这个$y$上输出的预测记为：
$$
\hat{x} = f_\theta(y) \tag{1}
$$
在使用MSE时，我们希望在给定$y$的情况下，最小化**期望平方误差**：
$$
\mathcal{L}_2(\hat{x}; y) = \mathbb{E}_{x\sim p_{\text{data}}(x|y)}\big[(x - \hat{x})^2\big]
\tag{2}
$$
把$y$看成参数，对某个固定的$y$，$\hat{x}$就是一个需要优化的标量。先在一维上推导，高维时每个维度类似：
$$
\begin{aligned}
\mathcal{L}_2(\hat{x}; y)
&= \mathbb{E}\big[x^2 - 2x\hat{x} + \hat{x}^2\big] \\
&= \mathbb{E}[x^2] - 2\hat{x}\mathbb{E}[x] + \hat{x}^2
\end{aligned}
\tag{3}
$$
<!-- more -->
对$\hat{x}$求导：
$$
\frac{\partial \mathcal{L}_2}{\partial \hat{x}}
= -2\mathbb{E}[x] + 2\hat{x}
= 2\big(\hat{x} - \mathbb{E}[x]\big)
\tag{4}
$$
令导数为0，可以得到最优预测：
$$
\hat{x}^\ast(y) = \mathbb{E}_{x\sim p_{\text{data}}(x|y)}[x]
\tag{5}
$$

我们可以得到MSE的本质是：对任意真实条件分布$p_{\text{data}}(x|y)$，在给定 $y$时，使用MSE训练的模型，最优情况下会输出
$$
\hat{x}(y) = \mathbb{E}[x|y] \tag{6}
$$
也就是**条件均值**。如果$p_{\text{data}}(x|y)$接近单峰高斯，这没有问题；但一旦分布有**离群点或者多峰结构**，就会出现下面的问题。

## MSE的问题
### 问题1：对离群点极其敏感
在很多生成或者恢复任务里，比如去噪、超分、CT重建等，数据里经常会出现少量异常样本：可能来自采集故障、配准误差、标注错误、金属伪影、极端剂量波动等。它们在数据集中占比很小，但误差往往很大。MSE的一个核心问题，就是对这类离群点（outliers）异常敏感。

考虑在某个固定的$y$下，真实的$x$服从一个混合分布：
$$
x|y \sim (1-\varepsilon)\cdot \mathcal{N}(0,1)+\varepsilon \cdot \mathcal{N}(M,1)
\tag{7}
$$
其中：
- 以概率$1-\varepsilon$，$x$在0附近波动（绝大多数“正常样本”）；
- 以概率$\varepsilon$，$x$在很远的$M$附近波动（很少量“极端样本”）；
- $\varepsilon$很小，比如$0.01$；$M$很大，比如100。

从分布图1上看，这就是“一个主要峰 + 一个很远但很小的次峰”：大部分概率质量集中在$0$附近，远端$M$附近只有一小撮样本。直觉上，一个“合理的预测”应该更忠于主峰，至少不应该被那1%的尾部样本显著改变。

<figure>
  <img src="./images/m1-n01-distribution.png" alt="混合分布示意图", width=500>
  <figcaption>图1 混合分布示意图</figcaption>
</figure>

由(5)式，MSE 的最优预测是条件均值：
$$
\begin{aligned}
\hat{x}^\ast(y)
&= \mathbb{E}[x|y] \\
&= (1-\varepsilon)\cdot 0 + \varepsilon\cdot M \\
&= \varepsilon M
\end{aligned}
\tag{8}
$$
如果$\varepsilon=0.01$、$M=100$，则
$$
\hat{x}^\ast(y) = 1
\tag{9}
$$

也就是说：尽管**99% 的真实样本都在0附近，仅仅$1\%$的远端样本，就把最优解从0拉到了1**。从概率质量的角度看，$1$并不是主峰位置，它甚至离主峰有明显偏移；但从平方误差的角度看，远端那一小撮样本因为残差巨大，会在目标函数里占据不成比例的权重，从而把最优解往它们的方向拽过去。这就是**均值被尾部强烈拉动**的典型例子。

这一点用梯度视角看更直接。对单个样本$(x, y)$，MSE的样本级损失是：
$$
\ell_2(x, \hat{x}) = (x - \hat{x})^2
$$
对$\hat{x}$求导：
$$
\frac{\partial \ell_2}{\partial \hat{x}}
= 2(\hat{x} - x)
\tag{10}
$$
可见梯度的绝对值与误差$x - \hat{x}$成正比，误差越大，梯度越大，**并且没有上界**。于是在训练时，少量极端样本会产生非常大的梯度，对参数更新的影响远远超过多数正常样本。

在生成任务里，这就意味着：
- 一小部分标注有问题或者采集有问题的图像，会让模型为“迁就它们”而改变整体行为；
- 轻微的数据污染或者偶发的标注错误，有可能对模型产生不成比例的巨大影响。

### 问题2：条件分布多峰下的「模糊平均解」
为了把“为什么会糊”说得更直观，我们考虑一个极端但非常典型的多峰例子：当同一个条件$y$对应多个合理的$x$时，典型的例子比如超分、补全、去噪等的细节不唯一，真实的条件分布$p_{\text{data}}(x|y)$往往就是多峰的。比如：
$$
p_{\text{data}}(x|y) = \tfrac{1}{2}\cdot\delta_{-1} + \tfrac{1}{2}\cdot\delta_{+1}
\tag{11}
$$
这代表两个完全合理的模式：一半样本的真值是$-1$；另一半样本的真值是$+1$。

从分布图2上看，这个条件分布的概率质量完全集中在$-1$和$+1$两个点上，中间的$0$处概率为$0$——也就是说，真实数据里从来没有出现过$x=0$。
<figure>
  <img src="./images/two-point-distribution.png" alt="两点分布示意图", width=500>
  <figcaption>图2 两点分布示意图</figcaption>
</figure>

但是MSE的最优解是条件均值：
$$
\hat{x}^\ast(y) = \mathbb{E}[x|y] = \frac{1}{2}(-1) + \frac{1}{2}(+1) = 0
\tag{12}
$$
这就暴露了MSE在多峰分布下的关键矛盾：它会把模型推向一个“平均值”，而这个平均值往往不是任何一个真实模式。在这个例子里，
**$0$从来没有在真实分布中出现过，它只是把$-1$和$+1$平均出来的结果**。

如果我们再从优化/风险曲线的角度看，会更清楚：把$\hat{x}$当作一个可调参数，画出期望损失随$\hat{x}$的变化：
<figure>
  <img src="./images/mse-two-peak-risk.png" alt="MSE在两点分布下的风险曲线", width=500>
  <figcaption>图3 MSE在两点分布下的风险曲线</figcaption>
</figure>

- 对MSE风险：
$$
\mathbb{E}_{x\sim p_{\text{data}}(x|y)}\big[(x - \hat{x})^2\big] = \tfrac{1}{2}(-1 - \hat{x})^2 + \tfrac{1}{2}(1 - \hat{x})^2 = \hat{x}^2 + 1 \tag{13}
$$
它是一条抛物线，只有一个全局最小点，严格落在$\hat{x} = 0$。因此，只要用MSE做风险最小化，训练就会稳定地把预测往0推，即使0根本不属于任何真实模态。

- 对L1风险：
$$
\mathbb{E}_{x\sim p_{\text{data}}(x|y)}\big[|x - \hat{x}|\big] = \tfrac{1}{2}|-1 - \hat{x}| + \tfrac{1}{2}|1 - \hat{x}| \tag{14}
$$
它的最小值不是一个点，而是一整段平底：任意 $\hat{x} \in [-1, 1]$ 都能达到同样的最小风险。这意味着在这种对称双峰条件分布下，L1 不会强迫模型必须输出一个特定的平均值，而是允许模型靠近某个模态或者处在模态之间的任意位置，因此它对多种合理真相的容忍度更高。

把这个一维例子映射回图像生成或者恢复任务就很直观了：当同一个$y$对应多种同样合理的细节，比如纹理、边缘位置、补全内容时，真实的$p_{\text{data}}(x|y)$就是多峰的。此时，MSE会执着于输出条件均值，相当于把多个模态叠加平均到一起，表现为边缘位置被抹平、纹理细节被平均化，也就是我们常说的“糊”。而L1在多峰条件下更不执着于唯一的均值解，因此在结构和边缘上往往更鲁棒。

## L1风险最小化
仍然在固定$y$的前提下，考虑用L1作为损失：
$$
\mathcal{L}_{\text{L1}}(\hat{x}; y)
= \mathbb{E}_{x\sim p_{\text{data}}(x|y)}\big[|x - \hat{x}|\big]
\tag{15}
$$

目标是找到最优预测$\hat{x}^\ast(y)$：
$$
\hat{x}^\ast(y) = \arg\min_{\hat{x}} \mathbb{E}\big[|x - \hat{x}|\big]
\tag{16}
$$

$|\cdot|$在$0$处不可导，我们用**次梯度(Subgradient)**。对单个样本：
$$
\frac{\partial}{\partial \hat{x}}|x - \hat{x}| =

\begin{cases}
-1, & x > \hat{x} \\
[-1, 1], & x = \hat{x} \\
+1, & x < \hat{x}
\end{cases}
\tag{17}
$$

对期望取次梯度，希望$0$属于这个“平均次梯度集合”。直观写法是：
$$
\mathbb{P}(x < \hat{x}^\ast(y)) \leq \tfrac{1}{2} \leq \mathbb{P}(x \leq \hat{x}^\ast(y))
\tag{18}
$$

这正是**中位数**的定义条件。所以我们可以得到**L1的本质是对任意真实条件分布$p_{\text{data}}(x|y)$，在给定$y$时，最小化 L1损失的最优预测$\hat{x}(y)$是$x$在该条件分布下的中位数，而不是均值。**

回到前面的两个例子：

- 在混合分布$x|y \sim (1-\varepsilon)\mathcal{N}(0,1) + \varepsilon\mathcal{N}(M,1)$中，只要$\varepsilon < 0.5$，分布的一半以上质量仍在$0$附近，**中位数仍然接近$0$**，不会像均值那样被拉到$\varepsilon M$。

- 在两点分布$p_{\text{data}}(x|y) = 0.5\delta_{-1} + 0.5\delta_{+1}$中，整个「中位数区间」在$[-1, +1]$之间，也就是说模型可以选择靠近任一个单点的峰，而不是被迫输出$0$这个“虚假均值”。这给了模型更大的灵活性去选择一个合理的模式，而不是被迫输出一个不存在的平均值。



这说明：**L1在离群点和多峰条件分布下，都比MSE更鲁棒。**


再看样本级梯度。仍然对单个样本$(x, y)$：
- 使用MSE时：
$$
  \ell_2(x, \hat{x}) = (x - \hat{x})^2
  \quad\Rightarrow\quad
  \frac{\partial \ell_2}{\partial \hat{x}} = 2(\hat{x} - x)
  \tag{19}
$$

- 使用L1时：
$$
  \ell_1(x, \hat{x}) = |x - \hat{x}|
  \quad\Rightarrow\quad
  \frac{\partial \ell_1}{\partial \hat{x}}
  = \operatorname{sign}(\hat{x} - x)=

  \begin{cases}
  +1, & \hat{x} > x \\
  -1, & \hat{x} < x
  \end{cases}
  \tag{20}
$$

对比可见：

- 对MSE而言，$|\partial \ell_2 / \partial \hat{x}| \propto |x - \hat{x}|$，**误差越大，梯度越大，没有上界**；
- 对L1而言，$|\partial \ell_1 / \partial \hat{x}|$始终是$1$，除了误差为$0$的点，**梯度幅值是有界的**。

这意味着：

> 在训练过程中，每个样本对参数更新的影响，不会因为该样本误差过大而变成“超级权重”；
> 少量极端错误样本不再拥有“无限放大的话语权”。


## 从拉普拉斯噪声到L1损失

最后，我们再回到和MSE那篇一模一样的概率建模框架。仍然假设有一个「真实映射」：
$$
x = f_\theta(y) + \epsilon \tag{21}
$$

只是这一次，不再假设$\epsilon$是高斯噪声，而是假设它是**拉普拉斯噪声**：
$$
\epsilon_i \sim \text{Laplace}(0, b) \tag{22}
$$

一维拉普拉斯分布的密度为：
$$
p(\epsilon_i)
= \frac{1}{2b}\exp\left(-\frac{|\epsilon_i|}{b}\right)
\tag{23}
$$

于是，每个维度的条件概率是：
$$
p(x_i|y; \theta)
= \frac{1}{2b}\exp\left(-\frac{|x_i - f_{\theta,i}(y)|}{b}\right)
\tag{24}
$$

若假设各个维度独立，则整幅图像的条件概率为：
$$
p(x|y; \theta)
= \prod_{i=1}^n \frac{1}{2b}
\exp\left(-\frac{|x_i - f_{\theta,i}(y)|}{b}\right)
\tag{25}
$$

对单个样本$(x_i, y_i)$取负对数：
$$
\begin{aligned}
-\log p(x_i|y_i; \theta) &= -\sum_{j=1}^n
\left[
\log\left(\frac{1}{2b}\right) + \frac{|x_{i,j} - f_{\theta,j}(y_i)|}{b}\right] \\
&= \sum_{i=1}^n\left[\log(2b) + \frac{1}{b}|x_{i,j} - f_{\theta,j}(y_i)|\right]
\end{aligned} \tag{26}
$$

对整个训练集求和，丢掉与$\theta$无关的常数项$\log(2b)$和缩放因子$\tfrac{1}{b}$，最终得到的优化目标是：
$$
\theta^\ast
= \arg\min_\theta \sum_{i=1}^N \sum_{j=1}^n
|x_{i,j} - f_{\theta,j}(y_i)|
= \arg\min_\theta \sum_{i=1}^n
|x_i - f_\theta(y_i)|_1
\tag{27}
$$

这正是 L1 损失函数：
$$
\mathcal{L}_{\text{L1}}(\theta)
= \sum_{i=1}^n |x_i - f_\theta(y_i)|_1 \tag{28}
$$

因此我们可以得到结论：在「误差$\epsilon$服从拉普拉斯分布」的假设下，对条件分布$p(x|y; \theta)$做极大似然估计，得到的训练目标正是**L1 损失**。