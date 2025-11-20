---
title: "DICOM图像处理"
createdAt: 2025-11-18
categories:
  - 医学图像
  - 动手实践
tags:
  - DICOM
  - 图像处理
---
DICOM（<u>D</u>igital <u>I</u>maging and <u>CO</u>mmunications in <u>M</u>edicine）是一种用于存储和传输医学影像的标准格式。DICOM文件由图像数据和头文件组成，图像数据是二维或三维的像素矩阵，头文件包含丰富的元数据信息，比如患者信息、扫描设备的信息，扫描参数等。下面以Mayo2016数据集中的DICOM图像为例，介绍如何进行基本的DICOM图像处理。

## 读取DICOM图像
我们可以使用Python的pydicom库来读取DICOM文件。以下是一个简单的示例代码：

```python
import pydicom
import numpy as np
import matplotlib.pyplot as plt
# 读取DICOM文件
ds = pydicom.dcmread('/path/to/dicom/file.dcm')
# 显示图像
image = ds.pixel_array
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.show()
```

<!-- more -->

显示结果如下：
<figure>
  <img src="./images/mayo_raw.png" alt="读取并显示DICOM图像示例", width=240>
  <figcaption>读取并显示DICOM图像示例</figcaption>
</figure>
CT值的单位是Hounsfield Unit (HU)，其计算公式为：
$$HU = 1000 \times \frac{\mu - \mu_{water}}{\mu_{water}-\mu_{air}}$$
其中，$\mu$是组织的线性衰减系数。

- 水的线性衰减系数$\mu_{water}=0.195cm^{-1}$；
- 空气的线性衰减系数$\mu_{air}=0$；
- 骨头的线性衰减系数$\mu_{bone}=0.78cm^{-1}$.

所以我们可以计算出不同组织的典型HU值：
$$HU_{water} = 1000 \times \frac{\mu_{water} - \mu_{water}}{\mu_{water} - \mu_{air}}=0$$
$$HU_{air} = 1000 \times \frac{\mu_{air} - \mu_{water}}{\mu_{water} - \mu_{air}}=-1000$$
$$HU_{bone} = 1000 \times \frac{0.78 - 0.195}{0.195 - 0} = 3000$$
因此，HU值的范围通常在-1000（空气）到+3000（骨骼）之间。

从上图可以看到，不做任何处理的DICOM图像对比度较低，细节不清晰。这是因为我们读取的DICOM图像的像素值有可能不是-1000到3000这个范围，通常是0~4096，这是我们常见到的像素值或者灰度值，我们需要将图像的像素值转换为HU值，才能更好地反映组织的密度信息。此外，还需要进行窗宽窗位调整，以增强图像的对比度。为此，我们需要从DICOM头文件中获取重缩放参数和窗宽窗位参数。

## 读取头文件信息
DICOM头文件中包含了丰富的元数据信息。我们可以通过pydicom库轻松访问这些信息。例如：

```python
ds = pydicom.dcmread('/path/to/dicom/file.dcm')
print("ds", ds)
```
输出结果包含了大量的头文件信息，我们这里挑选一些常用的字段进行展示：
```
# 文件元信息（0002组）
(0002,0000) File Meta Information Group Length  UL: 200
(0002,0001) File Meta Information Version      OB: b'\x00\x01'
...
# 图像信息（0008组）
(0008,0008) Image Type                          CS: ['ORIGINAL', 'PRIMARY', 'AXIAL', 'CT_SOM5 SPI']
...
(0008,0060) Modality                            CS: 'CT'
(0008,0070) Manufacturer                        LO: 'SIEMENS'
...
# 患者信息（0010组）
(0010,0010) Patient's Name                      PN: 'L096_FD_3'
(0010,0020) Patient ID                          LO: 'Anonymous'
# 扫描参数（0018组）
(0018,0015) Body Part Examined                  CS: 'CHEST'
(0018,0050) Slice Thickness                     DS: '3'
(0018,0060) KVP                                 DS: '120'
(0018,0090) Data Collection Diameter            DS: '500'
(0018,1020) Software Versions                   LO: 'syngo CT 2011A'
(0018,1100) Reconstruction Diameter             DS: '380'
(0018,1110) Distance Source to Detector         DS: '1085.6'
(0018,1111) Distance Source to Patient          DS: '595'
...
# 图像像素信息（0028组）
(0028,0002) Samples per Pixel                   US: 1
(0028,0004) Photometric Interpretation          CS: 'MONOCHROME2'
(0028,0010) Rows                                US: 512
(0028,0011) Columns                             US: 512
(0028,0030) Pixel Spacing                       DS: [0.7421875, 0.7421875]
...
(0028,1050) Window Center                       DS: [40, -600]
(0028,1051) Window Width                        DS: [400, 1500]
(0028,1052) Rescale Intercept                   DS: '-1024'
(0028,1053) Rescale Slope                       DS: '1'
(0028,1054) Rescale Type                        LO: 'HU'
(0028,1055) Window Center & Width Explanation   LO: ['WINDOW1', 'WINDOW2']
...
# 像素数据（7FE0组）
(7FE0,0010) Pixel Data                          OW: Array of 524288 elements
```
下面对这些常用字段进行简单介绍：
```
# 文件元信息（0002组）
(0002,0000) 文件元信息组长度 (UL)：文件元信息头的长度，通常为固定值，表示文件元信息的大小。
(0002,0001) 文件元信息版本 (OB)：文件元信息的版本，通常为'00 01'，表示当前版本。
...
# 图像信息（0008组）
(0008,0008) 图像类型 (CS)： 图像类型信息，包括'原始图像'、'主要'、'轴向'、'CT_SOM5 SPI'等。
...
(0008,0060) 模态 (CS)：模态类型，表示成像设备的类型，如CT、MR等。
(0008,0070) 制造商 (LO)：设备制造商名称，如'SIEMENS'。
...
# 患者信息（0010组）
(0010,0010) 患者姓名 (PN)：患者的姓名，如'L096_FD_3'。
(0010,0020) 患者ID (LO)：患者的唯一标识符，这里为'Anonymous'，表示匿名化的患者数据。
# 扫描参数（0018组）
(0018,0015) 检查部位 (CS)：检查的身体部位，如'CHEST'。
(0018,0050) 切片厚度 (DS)：切片的厚度，单位为毫米，例如'3'表示3mm厚。
(0018,0060) 管电压 (KVP) (DS)：X射线管的管电压（kVp），例如'120'表示120 kV。
(0018,0090) 数据采集直径 (DS)：数据采集的直径，如'500'。
...
(0018,1100) 重建直径 (DS)：图像重建的直径，如'380'。
(0018,1110) 源到探测器距离 (DS)：X射线源到探测器的距离，如'1085.6'。
(0018,1111) 源到患者距离 (DS)：X射线源到患者的距离，如'595'。
...
# 图像像素信息（0028组）
(0028,0002) 每像素的样本数 (US)：每个像素的样本数，通常为1表示单通道图像。
(0028,0004) 光度解释 (CS)：图像的光度解释方式，例如'MONOCHROME2'表示黑白图像。
(0028,0010) 行数 (US)：图像的行数，如512。
(0028,0011) 列数 (US)：图像的列数，如512。
(0028,0030) 像素间距 (DS)：像素间的物理距离，如[0.7421875, 0.7421875]毫米。
...
(0028,1050) 窗口中心 (DS)：窗位，例如[40, -600]。
(0028,1051) 窗口宽度 (DS)：窗宽，例如[400, 1500]。
(0028,1052) 重缩放截距 (DS)：重缩放截距，例如'-1024'。
(0028,1053) 重缩放斜率 (DS)：重缩放斜率，例如'1'。
(0028,1054) 重缩放类型 (LO)：重缩放类型，例如'HU'。
(0028,1055) 窗口中心和宽度说明 (LO)：窗口中心和宽度的说明，例如['WINDOW1', 'WINDOW2']。
...
# 像素数据（7FE0组）
(7FE0,0010) 像素数据 (OW)：像素数据数组，例如包含524288个元素。
```

## 图像处理
DICOM图像通常需要进行一些处理才能更好地显示和分析。常见的处理步骤包括重缩放和窗宽窗位调整。下面是一个简单的图像处理示例代码：
```python
import pydicom
import numpy as np
import matplotlib.pyplot as plt

# 读取DICOM文件
ds = pydicom.dcmread('/path/to/dicom/file.dcm')
# 获取图像数据
image = ds.pixel_array.astype(np.float32)
# 获取重缩放参数
intercept = ds.RescaleIntercept
slope = ds.RescaleSlope
# 重缩放图像
image = image * slope + intercept
# 获取窗宽窗位参数
window_center = ds.WindowCenter[1]
window_width = ds.WindowWidth[1]
# 应用窗宽窗位调整
min_window = window_center - window_width / 2
max_window = window_center + window_width / 2
image = np.clip(image, min_window, max_window)
# 归一化到0-255
image = (image - min_window) / (max_window - min_window) * 255.0
image = image.astype(np.uint8)
# 显示处理后的图像
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.show()
```
处理后的图像显示效果如下：
<figure>
  <img src="./images/mayo_processed.png" alt="处理后的DICOM图像示例" width=240>
  <figcaption>处理后的DICOM图像示例</figcaption>
</figure>

通过上述步骤，我们可以读取DICOM图像及其头文件信息，并对图像进行基本的处理，以便更好地显示和分析医学影像数据。