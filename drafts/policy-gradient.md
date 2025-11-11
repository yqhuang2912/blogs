---
title: "从策略梯度到REINFORCE算法"
createdAt: 2025-11-11
categories:
  - 技术分享
tags:
  - 数学推导
  - 强化学习
---

基于策略的强化学习(Policy-based RL)算法的核心本质是一个不确定的环境中，学习出一种行为方式（策略），使得Agent按照这种行为方式和环境交互的时候可以得到环境给予的最大回报。

## 策略到底是个啥？
策略就是Agent的行为规律，当Agent的处在一个确定的状态$s$的时候，需要决定执行什么动作$a$，选择的依据就是策略。

最一般的情况，Agent不一定每次都要做出同样的动作，因为有时候探索是必要的，所以策略就成了一个条件概率分布
$$\pi(a|s) = P(A_t=a|S_t=s) \tag 1$$
上面的公式表示，在状态$s$下，Agent选择动作$a$的概率是多少。由于策略是一个概率分布，所以它是可以被参数化的，通常我们用$\theta$来表示策略的参数，那么策略就可以写成$\pi_{\theta}(a|s)$。

<!-- more -->

## 为什么要关注轨迹？
因为我们的行为有后果，前面选的动作会影响后面的状态和奖励，所以我们需要关注Agent在环境中运行的一整条轨迹（trajectory）：
$$\tau = \{s_0,a_0,r_0,s_1,a_1,r_1,\cdots,s_T\} \tag 2$$
轨迹$\tau$表示Agent从初始状态$s_0$开始，选择动作$a_0$，得到奖励$r_0$，进入状态$s_1$，然后选择动作$a_1$，得到奖励$r_1$，进入状态$s_2$，一直持续到终止状态$s_T$。即：
$$s_0 \xrightarrow[a_0]{r_0} s_1 \xrightarrow[a_1]{r_1} s_2 \cdots s_T \tag 3$$

在Agent执行完一条轨迹$\tau$之后，我们可以计算出这条轨迹的累积回报（cumulative reward）：
$$R(\tau) = \sum_{t=0}^{T-1} \gamma^t r_t \tag 4$$
其中，$\gamma \in [0,1]$是折扣因子（discount factor），用于衡量未来奖励的重要性。

但是在一个有随机性的世界中，Agent即使使用相同的策略，在最终也可能会执行出不同的轨迹，所以**不能只看一次结果，要看平均表现**。在数学上，随机变量的平均表现可以用期望（Expectation）来表示：
$$J(\theta) = \mathbb{E}_{\tau \sim P_{\theta}(\tau)}[R(\tau)] \tag 5$$

这个式子代表了在这种策略下，平均能拿到多少回报。其中$P_{\theta}(\tau)$是Agent在环境中可能执行出的所有轨迹的分布，它表示在策略参数为$\theta$的情况下，Agent执行出轨迹$\tau$的概率。

## 强化学习的目标
强化学习的目标就是找到一组最优的策略参数$\theta^*$，使得期望回报$J(\theta)$最大化：
$$\theta^* = \arg\max_{\theta} J(\theta) \tag 6$$
这是一个优化问题，但是$J(\theta)$是一个复杂的函数，因为它包含了：
- 环境的动态变化
- 策略的随机性
- 状态转移的级联影响

所以我们无法直接求出$J(\theta)$的解析解。但是，我们可以通过计算$J(\theta)$的梯度$\nabla_{\theta} J(\theta)$，然后使用梯度上升法来迭代更新策略参数$\theta$，逐步逼近最优解。这就是策略梯度方法的核心思想。

## 策略梯度
使用期望的定义，我们可以将策略梯度写成下面的形式：
$$\nabla_{\theta} J(\theta) = \nabla_{\theta} \mathbb{E}_{\tau \sim P_{\theta}(\tau)}[R(\tau)] = \nabla_{\theta} \int P_{\theta}(\tau) R(\tau) d\tau \tag 7$$
由于xxx，满足莱布尼兹积分法则（Leibniz integral rule）的条件：
- 积分区间与参数无关
- 被积函数关于参数的导数连续
  
所以我们可以将梯度运算符移到积分符号内：
$$\nabla_{\theta} J(\theta) = \int \nabla_{\theta} P_{\theta}(\tau) R(\tau) d\tau \tag 8$$
为了计算$\nabla_{\theta} P_{\theta}(\tau)$，我们可以使用对数梯度技巧：
$$
\begin{aligned}
\nabla log f(x) &= \frac{1}{f(x)} \nabla_{\theta} f(x) \\
\Rightarrow \nabla_{\theta} f(x) &= f(x) \nabla log f(x)
\end{aligned}
$$
应用到$P_{\theta}(\tau)$上，我们有：
$$\nabla_{\theta} P_{\theta}(\tau) = P_{\theta}(\tau) \nabla_{\theta} \log P_{\theta}(\tau) \tag 9$$
将式(9)代入式(8)中，我们得到：
$$\begin{aligned}
\nabla_{\theta} J(\theta) &= \int P_{\theta}(\tau) \nabla_{\theta} \log P_{\theta}(\tau) R(\tau) d\tau \\
&= \mathbb{E}_{\tau \sim P_{\theta}(\tau)}[\nabla_{\theta} \log P_{\theta}(\tau) R(\tau)]
\end{aligned} \tag{10}$$
接下来，我们需要计算$\log P_{\theta}(\tau)$。根据轨迹的定义(式2)，轨迹的概率可以分解为初始状态分布、策略选择动作的概率和状态转移概率的乘积：
$$P_{\theta}(\tau) = P(s_0) \prod_{t=0}^{T-1} \pi_{\theta}(a_t|s_t) P(s_{t+1}|s_t,a_t) \tag{11}$$
取对数后，我们得到：
$$\log P_{\theta}(\tau) = \log P(s_0) + \sum_{t=0}^{T-1} \log \pi_{\theta}(a_t|s_t) + \sum_{t=0}^{T-1} \log P(s_{t+1}|s_t,a_t) \tag{12}$$
注意到初始状态分布$P(s_0)$和状态转移概率$P(s_{t+1}|s_t,a_t)$与策略参数$\theta$无关，所以它们的梯度为零。因此，我们有：
$$\nabla_{\theta} \log P_{\theta}(\tau) = \sum_{t=0}^{T-1} \nabla_{\theta} \log \pi_{\theta}(a_t|s_t) \tag{13}$$
将式(13)代入式(10)中，我们最终得到策略梯度的表达式：
$$\nabla_{\theta} J(\theta) = \mathbb{E}_{\tau \sim P_{\theta}(\tau)}\left[ \left( \sum_{t=0}^{T-1} \nabla_{\theta} \log \pi_{\theta}(a_t|s_t) \right) R(\tau) \right] \tag{14}$$
这个公式告诉我们，策略梯度可以通过采样轨迹，计算每个时间步的动作选择概率的对数梯度，并乘以整个轨迹的累积回报来估计.



