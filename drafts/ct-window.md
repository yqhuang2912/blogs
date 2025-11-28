---
title: "CT图像窗宽与窗位"
createdAt: 2025-11-18
categories:
  - 计算成像
tags:
  - 图像处理
---

在医学CT图像处理中，窗宽（Window Width, WW）和窗位（Window Level, WL）是两个非常重要的参数，用于调整图像的对比度和亮度，从而更好地显示不同组织结构的细节。

## 窗宽与窗位的定义
- 窗宽（WW）：表示CT图像中灰度级别的范围。较大的窗宽可以显示更多的灰度级别，适用于显示密度差异较大的组织（如肺部）；较小的窗宽则适用于显示密度差异较小的组织（如脑组织）。
- 窗位（WL）：表示CT图像灰度级别的中心值。通过调整窗位，可以改变图像的亮度，使得特定密度范围内的组织更加清晰可见。

<figure>
  <img src="./images/ct_window.png" alt="CT图像窗宽与窗位示意图", width=360>
  <figcaption>CT图像窗宽与窗位示意图</figcaption>
</figure>

<!-- more -->

## 窗宽与窗位的调整
调整窗宽和窗位通常通过以下公式实现：
$$HU_{min} = WL - \frac{WW}{2}$$
$$HU_{max} = WL + \frac{WW}{2}$$
$$I_{HU}[HU < HU_{min}] = HU_{min}$$
$$I_{HU}[HU > HU_{max}] = HU_{max}$$
其中，$HU_{min}$和$HU_{max}$分别表示CT图像中显示的最小和最大Hounsfield单位（HU）值。图像中低于$HU_{min}$的像素将显示为$HU_{min}$对应的灰度级别，高于$HU_{max}$的像素将显示为$HU_{max}$对应的灰度级别。

HU值和灰度级别的映射关系如下公式所示：
$$
I_{gray} = \frac{HU - HU_{min}}{HU_{max} - HU_{min}} \times 255
$$

## 常用窗宽与窗位设置
不同的组织结构通常使用不同的窗宽和窗位设置，以下是一些常用的设置：                         |

<table>
    <caption>表 1. CT窗宽和窗位设置</caption>
    <thead>
        <tr>
            <th>部位</th>
            <th>情况</th>
            <th>窗宽（WW, HU）</th>
            <th>窗位（WL, HU）</th>
        </tr>
    </thead>
    <tbody>
    <tr>
            <td rowspan="4">通用</td>
            <td>肺窗</td>
            <td>1500 ~ 2000</td>
            <td>-450 ~ -600</td>
        </tr>
        <tr>
            <td>纵膈窗</td>
            <td>250 ~ 350</td>
            <td>30 ~ 50</td>
        </tr>
        <tr>
            <td>骨窗</td>
            <td>1000 ~ 1500</td>
            <td>250 ~ 350</td>
        </tr>
        <tr style="border-bottom: 1px solid blue;">
            <td>软组织窗</td>
            <td>300 ~ 500</td>
            <td>40 ~ 60</td>
        </tr>
        <tr>
            <td rowspan="7">头颅</td>
            <td>脑组织窗</td>
            <td>80 ~ 100</td>
            <td>30 ~ 40</td>
        </tr>
        <tr>
            <td>垂体及蝶鞍区病变</td>
            <td>200 ~ 250</td>
            <td>45 ~ 50</td>
        </tr>
        <tr>
            <td>脑出血患者</td>
            <td>80 ~ 140</td>
            <td>30 ~ 50</td>
        </tr>
        <tr>
            <td>脑梗死患者</td>
            <td>~60（窄窗）</td>
            <td>—</td>
        </tr>
        <tr>
            <td>颌面部眼眶</td>
            <td>150 ~ 250</td>
            <td>30 ~ 40</td>
        </tr>
        <tr>
            <td>骨骼</td>
            <td>150 ~ 2000</td>
            <td>400 ~ 450</td>
        </tr>
        <tr style="border-bottom: 1px solid blue;">
            <td>喉颈部、鼻咽、咽喉部</td>
            <td>300 ~ 350</td>
            <td>30 ~ 50</td>
        </tr>
        <tr>
            <td rowspan="2">胸部</td>
            <td>纵隔窗</td>
            <td>250 ~ 350</td>
            <td>30 ~ 50</td>
        </tr>
        <tr style="border-bottom: 1px solid blue;">
            <td>肺窗</td>
            <td>1300 ~ 1700</td>
            <td>−600 ~ −800</td>
        </tr>
        <tr>
            <td rowspan="5">腹部</td>
            <td>常规窗</td>
            <td>300 ~ 500</td>
            <td>30 ~ 50</td>
        </tr>
        <tr>
            <td>肝/脾专窗</td>
            <td>100 ~ 200</td>
            <td>30 ~ 45</td>
        </tr>
        <tr>
            <td>肾脏</td>
            <td>200 ~ 300</td>
            <td>25 ~ 35</td>
        </tr>
        <tr>
            <td>胰腺</td>
            <td>300 ~ 350</td>
            <td>35 ~ 50</td>
        </tr>
        <tr style="border-bottom: 1px solid blue;">
            <td>胰腺窄窗</td>
            <td>120 ~ 150</td>
            <td>30 ~ 40</td>
        </tr>
        <tr>
            <td rowspan="2">脊柱及四肢</td>
            <td>脊椎旁软组织</td>
            <td>200 ~ 350</td>
            <td>35 ~ 45</td>
        </tr>
        <tr>
            <td>骨窗</td>
            <td>800 ~ 2000</td>
            <td>250 ~ 500</td>
        </tr>
    </tbody>
</table>

## 参考资料
1. 西安电子科技大学贾广老师: [CT图像与CT值](https://www.bilibili.com/video/BV1JJ411W7Fv?spm_id_from=333.788.videopod.episodes&vd_source=af1e89d4624a6f02ed73e2312d492273&p=33)
2. 知乎-鼎湖影像：[窗宽窗位，你应该知道的基础知识](https://zhuanlan.zhihu.com/p/482753403)
