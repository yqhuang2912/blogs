---
title: "生成模型系列（一）：从 VAE 出发"
createdAt: 2025-11-23
categories:
  - 人工智能
tags:
  - 生成模型
  - 概率图模型
  - VAE
---

## 判别模型 vs 生成模型：我们到底想学什么？

在监督学习里，我们最熟的是**判别模型（discriminative model）**，比如：
- 给定一张CT图像$x$，预测它是否有病灶$y$：学的是$p(y \mid x)$  
- 给定一段文本，预测情感标签：学的是$p(\text{label} \mid \text{text})$

这类模型关心的是「**输出标签**」，不关心数据本身是怎么「长出来的」。而**生成模型（generative model）**，则试图学的是**数据本身的分布** $p(x)$，甚至：
- 学习联合分布$p(x, y)$，从而既能做生成，也能做分类；  
- 或者学习一个条件分布$p(x \mid c)$，可以在条件$c$下生成样本（条件生成）。

## 几种主流的生成建模路径
粗略按「是否有显式的概率密度」划分，可以把流行的生成模型分成几类：

1. **显式密度模型（explicit density）**
   - 直接建模 $p_\theta(x)$ 或者 $p_\theta(x\mid z)$  
   - 通常配合最大似然训练（或其近似）
   - 代表：
     - 自回归模型（PixelCNN、GPT 等）
     - 变分自编码器（VAE）：$p_\theta(x, z) = p(z)p_\theta(x\mid z)$

2. **正则化变换模型（normalizing flows）**
   - 学一个可逆变换 $x = f_\theta(z)$，$z$ 服从简单分布（如高斯）
   - 通过变换的雅可比行列式精确算出 $p_\theta(x)$
   - 代表：RealNVP、Glow 等

3. **隐式生成模型（implicit model）**
   - 没有显式的 $p_\theta(x)$ 公式，只能通过采样间接定义分布
   - 多通过对抗训练、得分匹配、扩散过程来逼近数据分布
   - 代表：GAN、Diffusion Models

**VAE 属于第一类**：  
> 它假设存在一个潜变量 $z$，通过 $p_\theta(x\mid z)$ 来生成数据，  
> 再用「变分推断」去近似 $p_\theta(z\mid x)$。

---

## 3. 从概率建模视角看 VAE：引入潜变量

### 3.1 为何引入潜变量 $z$？

直接建模高维数据（比如 $512\times512$ 的医疗图像）的 $p(x)$ 通常非常难：

- 维度极高；
- 数据分布结构复杂（多模态、强非线性）；
- 很难直接写出一个参数化的、可积的 $p_\theta(x)$。

一个常见的思路是：**引入低维潜变量 $z$**：

- $z$ 可以理解为“生成这张图像时的一些隐含因素”：  
  - 在自然图像中：姿态、光照、类别、风格等；  
  - 在医学图像中：器官结构、病灶类型、成像条件等。
- 然后假设数据是「先采 $z$ 再生成 $x$」：

  $$
  z \sim p(z),\qquad
  x \sim p_\theta(x \mid z).
  $$

组合起来，得到联合分布：

$$
p_\theta(x, z) = p(z)\, p_\theta(x \mid z). \tag{3.1}
$$

边缘分布（对 $z$ 积分）就是我们真正想学的 $p_\theta(x)$：

$$
p_\theta(x) = \int p_\theta(x, z)\, dz = \int p(z)\, p_\theta(x\mid z)\, dz. \tag{3.2}
$$

> **直观理解**：  
> 我们假设「潜空间」里一切比较简单，  
> 把「复杂的高维数据」看成是潜空间里简单分布经过一个复杂非线性网络（decoder）“推出来”的结果。

---

### 3.2 生成模型的训练目标：最大似然

假设有数据集 $\{x^{(i)}\}_{i=1}^N$，最朴素的思想是用**最大似然估计（MLE）**来训练：

$$
\max_\theta \sum_{i=1}^N \log p_\theta(x^{(i)}). \tag{3.3}
$$

带入（3.2）：

$$
\log p_\theta(x^{(i)})
= \log \int p(z)\, p_\theta(x^{(i)} \mid z)\, dz. \tag{3.4}
$$

问题来了：  
> 这个积分通常**既高维又没有解析解**，而且里面有非线性神经网络，很难直接算出 $\log p_\theta(x)$。

于是，我们需要**两个东西**：

1. 一种方式来近似这个积分 / 对数似然；
2. 一种方式来近似后验 $p_\theta(z\mid x)$，因为它也会经常出现。

这就自然把我们带到了：**变分推断（variational inference）**。

---

## 4. 后验推断的难点：为什么要「变分」？

在潜变量模型中，贝叶斯后验是：

$$
p_\theta(z \mid x)
= \frac{p_\theta(x, z)}{p_\theta(x)}
= \frac{p(z)p_\theta(x\mid z)}{\int p(z')p_\theta(x\mid z')\, dz'}. \tag{4.1}
$$

- 分子好算（prior + likelihood）；
- 分母是我们刚才说的那坨难算的积分 $p_\theta(x)$。

这意味着：

- 后验 $p_\theta(z\mid x)$ 一般**没有解析形式**；
- 连归一化常数都不知道，很难直接从中采样或计算期望。

> 但 VAE 又非常想要这个后验——  
> 因为我们希望通过「编码器」把 $x$ 映射到潜空间 $z$，这个过程本质上就是在近似 $p_\theta(z\mid x)$。

于是，VAE 做了一件经典的事：  
> 用一个**可控、可微、参数化的网络** $q_\phi(z\mid x)$，去近似真实后验 $p_\theta(z\mid x)$。  
> 这就是「变分分布」或「变分后验」。

---

## 5. VAE 的核心：从 log p 到变分下界（ELBO）

接下来是 VAE 中最重要、也是最经典的推导步骤：  
**把 $\log p_\theta(x)$ 写成一个「下界 + KL」的形式**。

### 5.1 插入变分分布 $q_\phi(z\mid x)$

从边缘似然开始：

$$
\log p_\theta(x)
= \log \int p_\theta(x, z)\, dz. \tag{5.1}
$$

乘上 1（“乘除同一个东西”）：

$$
\begin{aligned}
\log p_\theta(x)
&= \log \int p_\theta(x, z)\, dz \\
&= \log \int 
\frac{p_\theta(x, z)}{q_\phi(z\mid x)}\,
q_\phi(z\mid x)\, dz. \tag{5.2}
\end{aligned}
$$

把这个积分解释为在 $q_\phi(z\mid x)$ 下的期望：

$$
\log p_\theta(x)
= \log \mathbb{E}_{z\sim q_\phi(z\mid x)}
\left[
  \frac{p_\theta(x, z)}{q_\phi(z\mid x)}
\right]. \tag{5.3}
$$

### 5.2 用 Jensen 不等式得到下界（ELBO）

对 $\log(\cdot)$ 使用 Jensen 不等式：

$$
\log \mathbb{E}[Y] \ge \mathbb{E}[\log Y].
$$

令
$$
Y = \frac{p_\theta(x, z)}{q_\phi(z\mid x)},
$$
得到：

$$
\begin{aligned}
\log p_\theta(x)
&= \log \mathbb{E}_{q_\phi}
\left[
  \frac{p_\theta(x, z)}{q_\phi(z\mid x)}
\right] \\
&\ge \mathbb{E}_{z\sim q_\phi(z\mid x)}
\left[
  \log \frac{p_\theta(x, z)}{q_\phi(z\mid x)}
\right]. \tag{5.4}
\end{aligned}
$$

记右边为：

$$
\mathcal{L}(\theta,\phi;x)
:= \mathbb{E}_{z\sim q_\phi(z\mid x)}
\left[
  \log \frac{p_\theta(x, z)}{q_\phi(z\mid x)}
\right]. \tag{5.5}
$$

这就是著名的 **变分下界（Evidence Lower BOund, ELBO）**：

$$
\boxed{
\log p_\theta(x) \ge \mathcal{L}(\theta,\phi;x)
}
$$

VAE 的训练目标就是：  
> 选择 $\theta,\phi$，让 **ELBO 尽可能大**。  
> ELBO 越大，说明：
> - 一方面 $p_\theta(x)$ 越大（模型越能解释数据），  
> - 另一方面变分后验 $q_\phi(z\mid x)$ 越接近真实后验。

---

### 5.3 把 ELBO 拆成「重构项 - KL 项」

对 $\mathcal{L}(\theta,\phi;x)$ 做一点代数处理：

先把 $p_\theta(x,z)$ 展开：

$$
p_\theta(x,z) = p(z) p_\theta(x\mid z). \tag{5.6}
$$

代入 (5.5)：

$$
\begin{aligned}
\mathcal{L}(\theta,\phi;x)
&= \mathbb{E}_{q_\phi}
\left[
  \log p(z) + \log p_\theta(x\mid z)
  - \log q_\phi(z\mid x)
\right] \\
&= \mathbb{E}_{q_\phi}
    \big[\log p_\theta(x\mid z)\big]
  + \mathbb{E}_{q_\phi}
    \big[\log p(z) - \log q_\phi(z\mid x)\big]. \tag{5.7}
\end{aligned}
$$

注意第二项：

$$
\mathbb{E}_{q_\phi}
\big[\log p(z) - \log q_\phi(z\mid x)\big]
= - \mathbb{E}_{q_\phi}
\left[
  \log \frac{q_\phi(z\mid x)}{p(z)}
\right]
= - \mathrm{KL}\big(q_\phi(z\mid x)\,\|\,p(z)\big). \tag{5.8}
$$

于是 ELBO 可以写成非常常见的形式：

$$
\boxed{
\mathcal{L}(\theta,\phi;x)
=
\underbrace{
\mathbb{E}_{z\sim q_\phi(z\mid x)}
  \big[\log p_\theta(x\mid z)\big]
}_{\text{重构项}}
-
\underbrace{
\mathrm{KL}\big(q_\phi(z\mid x)\,\|\,p(z)\big)
}_{\text{正则项：变分后验逼近先验}}
} \tag{5.9}
$$

这就是我们在 VAE 代码里常见的 loss 结构：

- **重构 loss**：让 decoder 在采样的 $z$ 上尽量把 $x$ 复原出来；
- **KL loss**：鼓励 $q_\phi(z\mid x)$ 不要偏离 prior 太远，避免潜空间乱跑。

---

### 5.4 另一种等价形式：log p = ELBO + KL

还有一个很重要、但常被忽略的等价式：

从一开始的：

$$
\log p_\theta(x)
= \mathcal{L}(\theta,\phi;x)
+ \mathrm{KL}\big(q_\phi(z\mid x)\,\|\,p_\theta(z\mid x)\big). \tag{5.10}
$$

这个式子可以从 (5.5) 再做一轮推导得到，这里直接给结论和直观解释：

- $\log p_\theta(x)$：固定不动（对 $\phi$ 来说是常数）；
- $\mathcal{L}(\theta,\phi;x)$：我们的优化目标；
- $\mathrm{KL}(q_\phi \| p_\theta)$：变分后验和真实后验的差距。

因为 KL 总是非负的：

$$
\mathrm{KL}\big(q_\phi \| p_\theta\big) \ge 0,
$$

所以：

- ELBO 是一个下界：$\mathcal{L} \le \log p_\theta(x)$；
- 当 $q_\phi(z\mid x) = p_\theta(z\mid x)$ 时，KL = 0，**下界恰好等于真实的 log-likelihood**。

> 这解释了「变分」这个名字：  
> 我们在所有可选的 $q_\phi(z\mid x)$ 中做“变分”，  
> 寻找一个让 ELBO 最大的 $q_\phi$，  
> 也就是让 KL 最小、最接近真实后验的那一个。

---

## 6. VAE 中的重参数化：为下一步做个铺垫

到这里，我们已经完成了 VAE 的**核心数学结构**：

1. 定义潜变量生成模型 $p_\theta(x,z)$；
2. 引入变分后验 $q_\phi(z\mid x)$；
3. 推到 ELBO：
   $$
   \mathcal{L} = \mathbb{E}_{q_\phi}[\log p_\theta(x\mid z)]
   - \mathrm{KL}(q_\phi \,\|\, p(z)).
   $$

接下来实际训练 VAE 时，会遇到一个关键问题：

> **这个 $\mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]$ 怎么反向传播？**

- 它是一个关于 $z$ 的期望；
- 而 $z$ 又是从 $q_\phi(z\mid x)$ 里“随机采样”的；
- 采样操作本身不可导。

**重参数化技巧（reparameterization trick）**干的事就是：

- 把采样 $z \sim q_\phi(z\mid x)$  
  改写成「先采一个固定分布的噪声 $\varepsilon$，再做一个可微变换 $z = g_\phi(\varepsilon,x)$」；
- 这样整个 ELBO 就可以写成对 $\varepsilon$ 的期望：
  $$
  \mathcal{L}(\theta,\phi;x)
  = \mathbb{E}_{\varepsilon\sim p_0}
  \big[\log p_\theta(x\mid g_\phi(\varepsilon,x))\big]
  - \mathrm{KL}_\text{analytic},
  $$
  然后就可以直接用普通的 backprop 来更新 $\theta,\phi$ 了。

在高斯 VAE 中，这个重参数化具体就是那句经典的：

$$
z = \mu_\phi(x) + \sigma_\phi(x)\odot \varepsilon,
\quad \varepsilon \sim \mathcal{N}(0, I).
$$

---

## 7. 小结 & 下篇预告

这一篇我们做了几件事：

1. 从「**生成模型 vs 判别模型**」开始，定位了 VAE 所属的方向；
2. 用潜变量模型的视角，写出了：
   - 联合分布 $p_\theta(x,z)$；
   - 边缘似然 $\log p_\theta(x)$；
3. 指出后验 $p_\theta(z\mid x)$ 的不可解性，动机地引出变分后验 $q_\phi(z\mid x)$；
4. 一步一步推到 VAE 的 ELBO，并给出了两种等价形式：
   - 重构项 - KL 项；
   - log p = ELBO + KL；
5. 预告了 VAE 中的关键实现问题：**期望上的梯度估计**，也就是重参数化要解决的核心。

> 在下一篇「生成模型系列（二）：VAE 中的重参数化与实现细节」里，我们会：
> - 站在 VAE 的角度，系统梳理重参数化技巧；
> - 写出高斯 VAE 中的完整数学推导；
> - 再给出一份带注释的 PyTorch 代码，把推导和代码一一对应起来。

