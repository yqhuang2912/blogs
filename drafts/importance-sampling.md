---
title: "重要性采样与Off-Policy强化学习"
createdAt: 2025-11-13
categories:
  - 数学研究
  - 人工智能
tags:
  - 数学推导
  - 强化学习
  - 蒙特卡洛
---

## 什么是重要性采样？

**重要性采样（Importance Sampling, IS）** 是一种经典的**蒙特卡洛方法**，主要用于在无法直接从目标分布采样，或者直接采样效率低下的情况下，**估计**某个函数在目标分布下的**期望值**。

它的核心思想非常直观：当我们想计算函数$f(x)$在目标分布$p(x)$下的期望$E_{x\sim p(x)}[f(x)]$时，如果直接从$p(x)$采样很困难（例如$p(x)$形式复杂），或者$f(x)$的高值区域在$p(x)$中出现的概率极低（即「稀有事件」），我们可以转而从另一个更容易采样、或更能「聚焦」于重要区域的**辅助分布**$q(x)$中抽取样本。

为了弥补从$q(x)$而非$p(x)$采样所带来的偏差，我们需要对每个样本进行「加权修正」。这个权重被称为**重要性权重（Importance Weight）**:
$$
w(x) = \frac{p(x)}{q(x)} \tag{1}
$$
它反映了：**在原分布$p(x)$下这个样本「应该有多重要」，与在$q(x)$下它的「被采到的概率」之间的比例关系。**

<!-- more -->

> **直观理解**
> 
> 想象你要估算一个大湖里某种稀有鱼类的平均长度。
> 
> - **直接采样$p(x)$**：如果你随机在整个湖里撒网，可能捞一天也捕不到几条，因为这种鱼很稀有，大部分时间和空网都在计算「0贡献」，效率极低。
> 
> - **重要性采样$q(x)$**：你请教了老渔夫，得知这种鱼喜欢聚集在深水芦苇区，于是你专门去芦苇区撒网，这就是你的建议分布$q(x)$。
> 
> - **权重修正**：你在芦苇区捕到了很多鱼，但不能直接用这些鱼的平均长度代表全湖的水平，因为你只去了鱼多的地方，有偏差。你需要根据「芦苇区面积占全湖的比例」以及「鱼在芦苇区的密度 vs 全湖密度」来调整计算。
> 
> 这就是重要性采样：**有策略地重点采样，再通过数学加权还原真相**。

重要性采样的基本步骤归纳如下：
1.  **设计分布**：找到一个合适的**建议分布（Proposal Distribution）**$q(x)$，它应该覆盖目标分布$p(x)$的重要区域，即$p(x)f(x)$较大的区域。
2.  **集中采样**：从$q(x)$中抽取样本$x_1, ..., x_N$。
3.  **加权修正**：计算每个样本的重要性权重，并进行加权平均。这确保了最终的估计在统计学上是**无偏**的，就像我们直接从$p(x)$采样一样。

## 重要性采样的数学原理

在数学上，我们的目标是估计函数$f(x)$在分布$p(x)$下的期望：
$$ \mathbb{E}_{x\sim p(x)}[f(x)] = \int f(x) p(x) \, dx \tag{2}$$

当直接对$p(x)$积分或采样困难时，我们引入建议分布$q(x)$。为了保证数学上的合法性，要求$q(x)$的支撑集（Support）必须覆盖$p(x)$的支撑集，即当$p(x)f(x) \neq 0$时，必须满足$q(x) > 0$。

利用恒等变换，将积分重写为：
$$ \mathbb{E}_{x\sim p(x)}[f(x)] = \int f(x) \frac{p(x)}{q(x)} q(x) \, dx \tag{3}$$

这个变换非常巧妙，它将问题转化为：在分布 $q(x)$ 下，计算新函数 $f(x) \frac{p(x)}{q(x)}$ 的期望：
$$ \mathbb{E}_{x\sim p(x)}[f(x)] = \mathbb{E}_{x\sim q(x)}\left[ f(x) \frac{p(x)}{q(x)} \right] \tag{4}$$

其中，比值 $w(x) = \frac{p(x)}{q(x)}$ 即为**重要性权重**。

在实际计算中，我们使用蒙特卡洛近似（Monte Carlo Approximation）：
1.  从$q(x)$中采样$N$个样本$\{x_i\}_{i=1}^N$。
2.  计算估计值：
$$ \hat{\mathbb{E}} = \frac{1}{N} \sum_{i=1}^N f(x_i) \frac{p(x_i)}{q(x_i)} = \frac{1}{N} \sum_{i=1}^N f(x_i) w(x_i) \tag{5}$$

可以证明，上面的估计是**无偏的**：
$$
\mathbb{E}_{x_1,\dots,x_N \sim q(x)} \left[ \hat{\mathbb{E}}_{x\sim p(x)}[f(x)] \right]
= \mathbb{E}_{x\sim p(x)}[f(x)] \tag{6}
$$
> **⚠️ 注意方差问题**
> 
> 虽然估计是无偏的，但**方差**可能非常大。如果$q(x)$选择不当，例如$p(x)$很大时$q(x)$很小，权重$w(x)$会变得巨大，导致估计值剧烈波动。因此，AI应用中选择合适的$q(x)$至关重要。


## 强化学习中的Off-Policy评估与策略优化
这是重要性采样在AI中最著名的应用。在强化学习中，一个基本的目标是评估某个策略$\pi_{\theta}$的好坏，使用如下的目标函数：
$$ J(\theta) = \mathbb{E}_{\tau \sim P(\tau|\pi_{\theta})} [R(\tau)] \tag{7}$$
其中：
- $\tau=(s_0,a_0, s_1, a_1, ..., s_T, a_T)$是一条轨迹；
- $R(\tau)$是整条轨迹的累计奖励；
- $P(\tau|\pi_{\theta})$是轨迹在策略$\pi_{\theta}$下的分布；

在**On-Policy**强化学习算法之中，例如原始的Policy Gradient，用于更新参数$\theta$的数据是必须是由当前策略$\pi_{\theta}$产生的，一旦我们更新了策略，参数变为$\theta^{'}$，旧策略$\pi_{\theta}$采集的数据就「过期」了，必须丢弃，因为数据分布变了。这导致由于需要频繁与环境交互，训练效率极低。

为了提高效率，我们希望数据可以被**重复利用**，这就引入了**Off-Policy**学习的概念。Off-Policy允许我们使用由旧策略$\pi_{old}$采集的数据来优化当前策略$\pi_{\theta}$。

这里的旧策略$\pi_{old}$和当前策略$\pi_{\theta}$分别对应于重要性采样中的建议分布$q(x)$和目标分布$p(x)$。因此我们可以看到在TRPO、PPO、GRPO等算法中，策略梯度的计算中包含了一个关键的比率：
$$ r_t(\theta) = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{old}(a_t|s_t)} \tag{8}$$
这个比率正是重要性权重，确保了我们在使用旧数据时，能够正确地估计新策略下的期望回报。
于是，强化学习的目标函数可以重写为：
$$ J(\theta) = \mathbb{E}_{\tau \sim P(\tau|\pi_{\theta})} [R(\tau)] = \mathbb{E}_{\tau \sim P(\tau|\pi_{\theta_{old}})} \left[R(\tau) \frac{P(\tau|\pi_{\theta})}{P(\tau|\pi_{\theta_{old}})} \right] \tag{9}$$

轨迹概率可以展开为：
$$ P(\tau|\pi_{\theta}) = P(s_0) \prod_{t=0}^{T-1} P(s_{t+1} | s_t, a_t) \pi_{\theta}(a_t|s_t) \tag{10}$$
类似地，
$$ P(\tau|\pi_{\theta_{old}}) = P(s_0) \prod_{t=0}^{T-1} P(s_{t+1} | s_t, a_t) \pi_{\theta_{old}}(a_t|s_t) \tag{11}$$
环境动力学和初始状态分布会约掉，得到轨迹级的重要性权重：
$$
\begin{aligned}
w(\tau) &= \frac{P(\tau|\pi_{\theta})}{P(\tau|\pi_{\theta_{old}})} \\
&=\frac{\prod_{t=0}^{T-1} P(s_{t+1} | s_t, a_t) \pi_{\theta}(a_t|s_t)}{\prod_{t=0}^{T-1} P(s_{t+1} | s_t, a_t) \pi_{\theta_{old}}(a_t|s_t)} \\
&= \prod_{t=0}^{T-1} \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}
\end{aligned} \tag{12}
$$

回顾我们之前的推导过的[策略梯度](./policy-gradient.html)公式：
$$ \nabla_{\theta} J(\theta) = \mathbb{E}_{\tau \sim P(\tau|\pi_{\theta})} \left[ \left (\sum_{t=0}^{T-1} \nabla_{\theta} \log \pi_{\theta}(a_t|s_t) \right)R(\tau) \right] \tag{13}$$
通过重要性采样，我们可以将上式改写为：
$$ \nabla_{\theta} J(\theta) = \mathbb{E}_{\tau \sim P(\tau|\pi_{\theta_{old}})} \left[\left(\sum_{t=0}^{T-1} \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} \nabla_{\theta} \log \pi_{\theta}(a_t|s_t) \right)R(\tau) \right] \tag{14}$$
这就是Off-Policy策略梯度的核心思想。通过引入重要性权重$w(\tau)$，我们能够**安全地使用旧策略采集的数据**来估计新策略的梯度，从而大大提高了数据利用率和训练效率。
