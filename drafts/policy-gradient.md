---
title: "从策略梯度到REINFORCE算法"
createdAt: 2025-11-11
categories:
  - 技术分享
tags:
  - 数学推导
  - 强化学习
---

基于策略的强化学习(Policy-based RL)算法的核心目标是找到一个好的策略，Agent使用这个“好策略”和环境交互的时候可以得到环境给与的最大回报。这个问题可以建模为：
$$\max_{\theta} J(\theta) = \max_{\theta} \mathbb{E}_{\tau \sim P_{\theta}(\tau)}[R(\tau)] \tag 1$$
其中:
- $\tau$是一条轨迹，是状态$s$，动作$a$和即时奖励$r$的集合$\{s_0,a_0,r_0,s_1,a_1,r_1,\cdots,s_T\}$，可以表示为：$s_0 \xrightarrow[a_0]{r_0} s_1 \xrightarrow[a_1]{r_1} \cdots s_T$；
- $P_{\theta}(\tau)$是Agent所能运行的所有可能的轨迹；
- $R(\tau)$是Agent在环境中按照轨迹$\tau$运行完所得到的累积回报。

<!-- more -->

根据期望的定义，将期望写成积分的形式如下面的公式所示：
$$
J(\theta) = \mathbb{E}_{\tau \sim P_{\theta}(\tau)}[R(\tau)] = \int P_{\theta}(\tau) R(\tau)d\tau \tag 2
$$
