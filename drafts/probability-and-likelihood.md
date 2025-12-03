---
title: "概率和似然"
createdAt: 2025-12-03
categories:
  - 数学研究
tags:
  - 概率分布
  - 极大似然
---
在概率论中，概率（Probability）和似然（Likelihood）是两个相关但是完全不同的概念。理解它们的区别对于统计推断和机器学习等领域非常重要。

## 概率和似然
- **概率**：描述的是**在已知模型参数$\theta$的情况下，观察到某个数据$D$的可能性**。例如：已知一枚硬币是公平的（这里的公平指的是正常的，没有被动过手脚的），那么抛一枚硬币，得到正面正面朝上的概率是就是$p(正面朝上|硬币公平)=0.5$。这里说的<u>模型的参数已知可以理解为投掷硬币的概率分布$P_\theta$是已知的</u>。概率的公式为：
$$p(D|\theta),\theta=\begin{cases}0.5 & D=正面朝上 \\ 0.5 & D=反面朝上 \end{cases} \tag{1}$$
其中，$D$表示观察到的数据，$\theta$表示模型的参数。那么，在已知$\theta$的情况下，我们可以计算抛10次硬币，得到7次正面朝上的概率为：
$$p(D|\theta) = \binom{10}{7} (0.5)^7 (0.5)^3 \tag{2}$$

- **似然**：描述的是**在已知数据$D$的情况下，模型参数$\theta$取某个值的可能性**。例如：假设我们抛了一枚硬币10次，结果得到7次正面朝上，3次反面朝上。那么，我们想知道这枚硬币是公平的（$\theta=0.5$）的可能性有多大，或者偏向正面朝上的（$\theta>0.5$）的可能性更大。计算的方式为：
$$\mathcal{L}(\theta|7正面,3反面) = \binom{10}{7} \theta^7 (1-\theta)^3 \tag{3}$$

<!-- more -->

从上面的公式(2)和(3)可以看出，概率和似然的数学表达式都可以写成：
$$p(D|\theta) = \binom{10}{7} \theta^7 (1-\theta)^{3} \tag{4}$$

因此，假设我们有一个概率模型，其参数用$\theta$表示，观测数据为：$D=\{x_1,x_2,\cdots,x_n\}$，假设观测数据之间相互独立，则似然可以定义为：
$$\mathcal{L}(\theta|D) := p(D|\theta) = \prod_{i=1}^{n} p(x_i|\theta) \tag{5}$$

**区别是概率是参数已知，数据未知，似然是观测数据已知，参数未知**。要注意的是，从严格的概率论的角度来看，似然函数并不是真正的概率，它是在已知数据$D$的前提下，评估参数$\theta$的**适应性（这个参数能不能很好的代表这个模型）**。换句话说，似然函数衡量的是不同的参数$\theta$的取值对于观测数据$D$的解释能力。

## 极大似然估计
那么该怎么提高评估参数的“适应性”，或者说找到一个参数$\hat{\theta}$，使得这个参数对于观测数据的解释能力最强呢？通俗来说，就是找到一个参数$\hat{\theta}$，使得观测数据在这个以$\hat{\theta}$为参数的模型下出现的概率最大，即：
$$
\hat{\theta} = \arg\max_{\theta} \mathcal{L}(\theta|D)=\arg\max_{\theta} p(D|\theta) \tag{6}
$$

这就是**极大似然估计(Maximum Likehood Estimation, MLE)**，它是一种经典的参数估计方法，其目标是找到参数$\hat{\theta}$使得似然函数最大。

在实际使用中，我们通常用对数似然函数来替换原始的似然函数，这是因为概率值通常很小，直接计算似然函数可能会导致数值下溢（Underflow），对数似然的定义如下：
$$
\log \mathcal{L}(\theta|D)=\log p(D|\theta)=\log\prod_{i=1}^{n}p(x_i|\theta)=\sum_{i=1}^{n}\log p(x_i|\theta) \tag{7}
$$
则对数似然的极大化问题可以表示为：
$$
\hat{\theta} = \arg\max_{\theta} \log \mathcal{L}(\theta|D)=\arg\max_{\theta} \sum_{i=1}^{n}\log p(x_i|\theta) \tag{8}
$$

对数似然函数的优点如下：
1. 将乘法转换为加法，避免数值下溢
2. 对数函数是单调递增的，所以最大化对数似然等价于最大化原始的似然函数，对于优化结果没有影响
3. 便于计算梯度和优化

如果我们在式(8)的右边除以$n$，则可以得到一个更加紧凑的形式：
$$
\begin{aligned}
\hat{\theta} &= \arg\max_{\theta} \frac{1}{n} \sum_{i=1}^{n}\log p(x_i|\theta) \\
&= \arg\max_{\theta} \mathbb{E}_{x \sim p_{data}}\left[\log p(x|\theta)\right]
\end{aligned}  \tag{9}
$$

对于式(9)，如果我们在最大化的目标前面加一个负号，将其转化为一个最小化问题，可以得到：
$$
\begin{aligned}
\hat{\theta} &= \arg\min_{\theta} -\mathbb{E}_{x \sim p_{data}}\left[\log p(x|\theta)\right] \\
&= \arg\min_{\theta} - \sum_{x} p(x)\log p(x|\theta)
\end{aligned}  \tag{10} 
$$
式(10)中的目标函数实际上是**交叉熵（Cross-Entropy）**，它衡量了真实数据分布$p_{data}$和模型分布$p(x|\theta)$之间的差异。因此，<span style="color:blue;">极大似然估计等价于最小化交叉熵</span>，这也是机器学习中常用的损失函数之一。

