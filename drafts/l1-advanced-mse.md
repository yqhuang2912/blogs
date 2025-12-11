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


在上一篇《**为什么生成模型要使用MSE？**》中，我们从概率建模的角度推导了一个结论：
> 如果假设误差:
> $$x = f_\theta(y) + \epsilon$$中的噪声$\epsilon$服从高斯分布，那么最大化似然就等价于最小化
> $$(x - f_\theta(y))^2$$
> 也就是 **MSE 损失**。

MSE看起来非常自然，也有漂亮的数学和物理背景：中心极限定理 + 最大熵。但在实际的生成模型中，比如图像去噪、超分、CT 重建、超分辨率等，很多人会发现：仅仅使用MSE会带来一些问题，比如图像发糊、细节被抹平、对少量异常样本极度敏感。

## MSE本质上逼近的是均值

我们先把条件$y$固定下来，只关注「在这个$y$下，$x$的真实分布是什么」。设真实数据来自$p_{\text{data}}(x|y)$，模型在这个$y$上输出的预测记为：
$$
\hat{x} = f_\theta(y) \tag{1}
$$
在使用MSE时，我们希望在给定$y$的情况下，最小化**期望平方误差**：
$$
\mathcal{L}_2(\hat{x}; y) = \mathbb{E}_{x\sim p_{\text{data}}(x\mid y)}\big[(x - \hat{x})^2\big]
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
\hat{x}(y) = \mathbb{E}[x|y]
$$
也就是**条件均值**。如果$p_{\text{data}}(x|y)$接近单峰高斯，这没有问题；但一旦分布有**重尾/离群点/多模态结构**，就会出现下面的问题。

## MSE的问题：
### 问题1：对离群点极其敏感

考虑在某个固定的$y$下，真实的$x$服从这样一个混合分布：
$$
x|y \sim (1-\varepsilon)\cdot \mathcal{N}(0,1)+\varepsilon \cdot \mathcal{N}(M,1)
\tag{6}
$$
解释一下：

* 以概率$1-\varepsilon$，$x$在0附近波动（绝大多数“正常样本”）；
* 以概率$\varepsilon$，$x$在很远的$M$附近波动（很少量“极端样本”）；
* $\varepsilon$很小，比如$0.01$；$M$很大，比如100。

由(5)式，MSE 的最优预测是条件均值：
$$
\begin{aligned}
\hat{x}^\ast(y)
&= \mathbb{E}[x|y] \\
&= (1-\varepsilon)\cdot 0 + \varepsilon\cdot M \\
&= \varepsilon M
\end{aligned}
\tag{6}
$$
如果$\varepsilon=0.01$、$M=100$，则
$$
\hat{x}^\ast(y) = 1
\tag{7}
$$

尽管 **99% 的真实样本都在0附近**，仅仅$1\%$的远端样本，就把最优解从0拉到了1。

这就是**均值被尾部强烈拉动**的典型例子。

从梯度的角度看也一样。对单个样本$(x, y)$，MSE的样本级损失是：
$$
\ell_2(x, \hat{x}) = (x - \hat{x})^2
$$
对 (\hat{x}) 求导：
$$
\frac{\partial \ell_2}{\partial \hat{x}}
= 2(\hat{x} - x)
\tag{8}
$$
误差$x - \hat{x}$越大，这个梯度的绝对值越大，**没有上界**。于是：

> 训练时，少量极端样本会产生非常大的梯度，
> 对参数更新的影响远远超过多数正常样本。

在生成任务里，这就意味着：

* 一小部分“标注脏/采集有问题”的图像，
  会让模型为“迁就它们”而改变整体行为；
* 轻微的数据污染 / 偶发的标注错误，
  有可能对模型产生不成比例的巨大影响。

---

## 3. MSE 的问题二：多模态条件分布下的「模糊平均解」

再看另一种常见情况：**多模态条件分布**。
还是在固定 (y) 的前提下，设真实分布是：

[
p_{\text{data}}(x\mid y) = \tfrac{1}{2},\delta_{-1} + \tfrac{1}{2},\delta_{+1}
\tag{9}
]

也就是说：

* 一半样本的真值是 (-1)；
* 另一半样本的真值是 (+1)。

这代表“**两个完全合理的模式**”：
比如，在某个视角下的图像，有一半人是张嘴、一半人是闭嘴；
或者在某个扫描条件下，结构有两种可能位置。

由(4)式，MSE 的最优解是条件均值：

[
\hat{x}^\ast(y)
= \tfrac{1}{2}(-1) + \tfrac{1}{2}(+1)
= 0
\tag{10}
]

**注意：0 从来没有在真实分布中出现过。**
它只是“两个合理值 (-1) 和 (+1) 的平均”。

放到图像上就是：

* 一部分图像的边缘在位置 A；
* 一部分图像的边缘在位置 B；
* 使用 MSE 时，模型在优化过程中会被迫靠近「平均的边缘位置」，
  结果就是 **边缘被抹平、纹理被平均化**——即我们常说的“糊”。

---

## 4. 从 MSE 的问题到我们想要的性质

到目前为止，我们已经看到：

1. **在有离群点 / 重尾误差时：**

   * MSE 的最优解（条件均值）会被少量极端样本强烈拉动；
   * 样本级梯度的绝对值与误差线性相关，离群点产生巨大梯度。

2. **在多模态分布下：**

   * MSE 让模型逼近“条件均值”，而“均值”常常不是任何一个真实模式；
   * 这直接表现为：图像变得平滑、缺乏锐利的结构和纹理。

因此，我们希望找到一种更合适的损失，使得：

* 对离群点**更鲁棒**，不会被少量极端样本支配；
* 在多模态情形下，**不要那么执着于“平均值”**，而是更忠于主流模式；
* 同时仍然保留良好的优化特性。

这就是引出 **L1 损失** 的自然契机。

---

## 5. L1 风险最小化：从 (|x - \hat{x}|) 出发

仍然在固定 (y) 的前提下，考虑用 L1 作为损失：

[
\mathcal{L}*1(\hat{x}; y)
= \mathbb{E}*{x\sim p_{\text{data}}(x\mid y)}\big[,|x - \hat{x}|,\big]
\tag{11}
]

目标是找到最优预测 (\hat{x}^\ast(y))：

[
\hat{x}^\ast(y)
= \arg\min_{\hat{x}} \mathbb{E}\big[,|x - \hat{x}|,\big]
\tag{12}
]

(|\cdot|) 在 0 处不可导，我们用**次梯度**。对单个样本：

[
\frac{\partial}{\partial \hat{x}}|x - \hat{x}|
==============================================

\begin{cases}
-1, & x > \hat{x} \
[-1, 1], & x = \hat{x} \
+1, & x < \hat{x}
\end{cases}
\tag{13}
]

对期望取次梯度，希望 0 属于这个“平均次梯度集合”。直观写法是：

[
\mathbb{P}(x < \hat{x}^\ast(y)) ;\le; \tfrac{1}{2} ;\le; \mathbb{P}(x \le \hat{x}^\ast(y))
\tag{14}
]

这正是**中位数（median）**的定义条件。

**结论 2（L1 的本质）：**

> 对任意真实条件分布 (p_{\text{data}}(x\mid y))，在给定 (y) 时，
> 最小化 L1 损失的最优预测 (\hat{x}(y)) 是
> **(x) 在该条件分布下的中位数**，而不是均值。

回到前面的两个问题：

* 在混合分布
  [
  x\mid y \sim (1-\varepsilon)\mathcal{N}(0,1) + \varepsilon\mathcal{N}(M,1)
  ]
  中，只要 (\varepsilon < 0.5)，分布的一半以上质量仍在 0 附近，
  **中位数仍然接近 0**，不会像均值那样被拉到 (\varepsilon M)。

* 在两点分布
  [
  p_{\text{data}}(x\mid y) = 0.5,\delta_{-1} + 0.5,\delta_{+1}
  ]
  中，整个「中位数区间」在 ([-1, +1]) 之间，
  也就是说模型可以选择靠近任一模态，而不是被迫输出 0。

这说明：

> L1 在**离群点**和**多模态条件分布**下，都比 MSE 更鲁棒。

---

## 6. 梯度视角：L1 的影响是「有界」的

再看样本级梯度。仍然对单个样本 ((x, y))：

* 使用 MSE 时：
  [
  \ell_2(x, \hat{x}) = (x - \hat{x})^2
  \quad\Rightarrow\quad
  \frac{\partial \ell_2}{\partial \hat{x}} = 2(\hat{x} - x)
  \tag{15}
  ]

* 使用 L1 时：
  [
  \ell_1(x, \hat{x}) = |x - \hat{x}|
  \quad\Rightarrow\quad
  \frac{\partial \ell_1}{\partial \hat{x}}
  = \operatorname{sign}(\hat{x} - x)
  ==================================

  \begin{cases}
  +1, & \hat{x} > x \
  -1, & \hat{x} < x
  \end{cases}
  \tag{16}
  ]

对比可见：

* 对 MSE 而言，(|\partial \ell_2 / \partial \hat{x}| \propto |x - \hat{x}|)，
  误差越大，梯度越大，没有上界；
* 对 L1 而言，(|\partial \ell_1 / \partial \hat{x}|) 始终是 1（除了误差为 0 的点），
  **梯度幅值是有界的**。

这意味着：

> 在训练过程中，每个样本对参数更新的影响，不会因为该样本误差过大而变成“超级权重”；
> 少量极端错误样本不再拥有“无限放大的话语权”。

这和上一节“中位数 vs 均值”的结论是一致的：L1 在统计意义上更鲁棒，在优化动力学上也更稳健。

---

## 7. 概率建模视角：拉普拉斯噪声 (\Rightarrow) L1 损失

最后，我们再回到和 MSE 那篇一模一样的概率建模框架。

仍然假设有一个「真实映射」：

[
x = f_\theta(y) + \epsilon
\tag{17}
]

只是这一次，不再假设 (\epsilon) 是高斯噪声，而是假设它是**拉普拉斯噪声**：

[
\epsilon_i \sim \text{Laplace}(0, b)
]

一维拉普拉斯分布的密度为：

[
p(\epsilon_i)
= \frac{1}{2b}\exp\left(-\frac{|\epsilon_i|}{b}\right)
\tag{18}
]

于是，每个维度的条件概率是：

[
p(x_i\mid y; \theta)
= \frac{1}{2b}\exp\left(-\frac{|x_i - f_{\theta,i}(y)|}{b}\right)
\tag{19}
]

若假设各个维度独立，则整幅图像的条件概率为：

[
p(x\mid y; \theta)
= \prod_{i=1}^D \frac{1}{2b}
\exp\left(-\frac{|x_i - f_{\theta,i}(y)|}{b}\right)
\tag{20}
]

对单个样本 ((x_n, z_n)) 取负对数：

[
\begin{aligned}
-\log p(x_n\mid z_n; \theta)
&= -\sum_{i=1}^D
\left[
\log\left(\frac{1}{2b}\right)

* \frac{|x_{n,i} - f_{\theta,i}(z_n)|}{b}
  \right] \
  &= \sum_{i=1}^D
  \left[
  \log(2b) + \frac{1}{b}|x_{n,i} - f_{\theta,i}(z_n)|
  \right]
  \end{aligned}
  \tag{21}
  ]

对整个训练集求和，丢掉与 (\theta) 无关的常数项 (\log(2b)) 和缩放因子 (\tfrac{1}{b})，最终得到的优化目标是：

[
\theta^\ast
= \arg\min_\theta \sum_{n=1}^N \sum_{i=1}^D
|x_{n,i} - f_{\theta,i}(z_n)|
= \arg\min_\theta \sum_{n=1}^N
|x_n - f_\theta(z_n)|_1
\tag{22}
]

这正是 L1 损失函数：

[
\mathcal{L}*{\text{L1}}(\theta)
= \sum*{n=1}^N |x_n - f_\theta(z_n)|_1
]

**结论 3（L1 的概率意义）：**

> 在「误差 (\epsilon) 服从拉普拉斯分布」的假设下，
> 对条件分布 (p(x\mid y; \theta)) 做极大似然估计，
> 得到的训练目标正是 **L1 损失**。

---

## 8. 小结：从 MSE 的缺陷走向 L1 的优势

把这一篇的内容压缩成一个「从 MSE 出发走到 L1」的路线图，就是：

1. **使用 MSE 时**：

   * 最小化的是 (\mathbb{E}[(x - \hat{x})^2])，
   * 最优预测是条件均值 (\hat{x}^\ast(y) = \mathbb{E}[x\mid y])；
   * 在离群点、重尾、多模态场景下，均值会被严重拉偏。

2. **MSE 面临的问题**：

   * 对少量极端样本非常敏感（均值和梯度都被远端样本支配）；
   * 在多模态条件分布下，输出“平均解”，导致图像模糊、细节被抹平。

3. **使用 L1 时**：

   * 最小化的是 (\mathbb{E}[|x - \hat{x}|])，
   * 最优预测是条件中位数（更鲁棒）；
   * 样本级梯度是 (\operatorname{sign}(\hat{x} - x))，幅值有界，不会被离群点“放大”。

4. **从概率建模看**：

   * 高斯噪声 (\epsilon\sim\mathcal{N}(0,\sigma^2I)) → 极大似然 ⇔ MSE；
   * 拉普拉斯噪声 (\epsilon\sim\text{Laplace}(0,b)) → 极大似然 ⇔ L1。

因此，当你在生成模型中观察到：

* 数据中存在明显的离群点或重尾误差；
* 输出存在多模态结构，MSE 导致结果偏“泛泛安全但很糊”；

从数学上看，**把损失从 MSE 换成（或至少部分换成） L1，是一种有坚实理论依据的选择**，而不是简单的“经验调参”。
