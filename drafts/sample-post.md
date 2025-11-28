---
title: "一次实验记录：自定义模型的损失分析"
createdAt: 2025-11-06
categories:
  - 工程实践
  - 人工智能
tags:
  - 训练
  - 日志
  - 可视化
  - 测试
summary:
  - type: p
    html: "快速记录一次模型训练实验的配置与关键指标。"
---
在这篇短文里，我们记录一次图像分类模型的训练过程。实验运行在单张 RTX 4090 上，总耗时 12 小时。

## 数据集准备

原始数据集包含 12 个类别，共 18,240 张样本。我们应用了随机裁剪、颜色抖动与 Mixup 增广。配合 224x224 的输入尺寸，总体扩增倍率约为 2.8 倍。![训练损失曲线](./images/chart.png "训练阶段的损失曲线")

> 训练前 20 个 epoch 使用余弦退火的 warmup 学习率调度，基础学习率设置为 3e-4。

## 指标汇总

<table>
    <caption>表 1. 指标汇总</caption>
    <thead>
        <tr>
            <th>指标</th>
            <th>数值</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Top-1 Accuracy</td>
            <td>87.4%</td>
        </tr>
        <tr>
            <td>Top-5 Accuracy</td>
            <td>96.1%</td>
        </tr>
        <tr>
            <td>F1 Score</td>
            <td>0.881</td>
        </tr>
    </tbody>
</table>


<figure>
  <img src="./images/cat.png" alt="训练阶段的损失曲线", width=360>
  <figcaption>图 1. 训练阶段的损失曲线</figcaption>
</figure>

<!-- more -->

## 训练过程
$$
\mathcal{L}\_{\text{GRPO}}=\mathbb{E}\_{q\sim P(Q), o \sim \pi\_{\theta\_{\text{old}}}(O|p)}\left[\frac{1}{|o|}\right]
$$

### 训练日志片段

```python
for epoch in range(start_epoch, total_epochs):
    stats = train_one_epoch(model, dataloader=train_loader, optimizer=optim)
    scheduler.step()
    if should_validate(epoch):
        evaluate(model, val_loader)
```

最终模型的 EMA 版本在验证集上提供了额外 0.6% 的准确率增益。

另外的测试代

```python
import torch
a = 1000
b = 2000
x = func(a+b) # test
```
