# YOLOv7网络架构研究及改进

# 1. 代码GITHUB网址：

https://github.com/threeban/yolov7-BiFPN

# 2. 动机：

近些年来，随着计算机领域的发展，计算机视觉在不同的应用场景下取得了许多重大的突破和创新，其中目标检测是计算机视觉领域中的一个重要技术，有着极高的应用价值。基于深度学习的目标检测算法近年来发展迅速，这类算法不需要手动设计特征，可以直接通过卷积神经网络自动学习图像特征，可以适用于更多场景，处理更加复杂的情况。

2020年，Tan等人[1]提出了一种加权双向特征金字塔网络(bi-directional feature pyramid network, BiFPN)，加权融合的方式和双向跨尺度连接结构能有效地利用不同尺度的特征，提高模型的表达能力和检测性能。他们还提出了一种混合缩放方法，可以均匀地对所有主干网络、特征网络和最终的预测网络的分辨率、深度和宽度进行缩放，在一定程度上提高了检测性能

2022年，Wang等人[2]改进ELAN（Efficient Layer Aggregation Networks）网络结构，提出扩展的高效层聚合网络E-ELAN，并加入RepConv重参数化卷积、辅助头检测等提升了YOLO模型的检测速度和精度

本课题基于深度学习技术，对多场景目标检测技术实用性开展研究，从单阶段目标检测算法出发，更加高效地利用图像信息，来应对更加复杂的检测场景，为诸多领域提供更加高效准确的目标检测技术。

# 3. 创新点：

如本研究拟在YOLOv7特征融合阶段引入BiFPN，替代简单的路径聚合网络 (Path Aggregation Network, PAN)和特征金字塔网络(Feature Pyramid Networks, FPN)，增强神经网络之间特征信息的传递，提升算法检测精度。

# 4. 方法：

## YOLOv7原理

YOLOv7是YOLO系列中的基本模型，YOLOv7具备三种基本模型，小模型为YOLOv7-tiny，标准大小模型YOLOv7和大模型YOLOv7-X。在处理速度和检测准确度方面，YOLOv7模型表现优异，在多个数据集的实验中超过了许多热门的目标检测器。与YOLOv5相比，YOLOv7的检测思路相似，其网络架构如图1所示。

![1718637671202](image/README/1718637671202.png)

图1 YOLOv7网络架构

YOLOv7网络模型可以分为主干网络(Backbone)、颈部网络(Neck)、检测头(Head)三部分。YOLOv7会先对输入的图像进行图像归一化操作以及数据增强处理，预处理可以对图像中的信息有选择地加强，使模型更好的进行分析。经过处理后的图像适用于模型的训练，可以通过内置的加载器读入模型开始训练或进行推理。YOLOv7的主干网络包含CBS、ELAN和MP-C3三种基础模块，基础模块会根据其在网络的不同位置进行通道数和分支数的调整以达到更好的效果。其中CBS模块如图2所示，由三部分组成。输入的数据会先经过卷积(Conv)提取特征值，然后使用批量样本归一化(BN)使特征值的传递更加稳定，最后经过激活函数SiLU增强特征值的表达，同时CBS结构在YOLOv7中根据卷积神经网络卷积核和步长的不同可以分为三种不同的CBS模块。

![](file:///C:/Users/steak/AppData/Local/Temp/msohtmlclip1/01/clip_image006.png)![1718637676056](image/README/1718637676056.png)

图2 CBS结构

ELAN如图3所示由多个CBS模块构成，使用了3个卷积核为1×1步长为1的CBS和4个卷积核为3×3步长为1的CBS。输出通道都是和输入通道保持一致的。在开始的两个分支的中，使用卷积核为1×1步长为1的CBS使通道数发生变化。而后续的几个CBS的输入通道与输出通道保持一致，并经过最后一个CBS输出所需的通道数。ELAN-4和ELAN-6设计思路相近，仅仅是cat的分支数不同，分别用于Backbone和Neck部分。

![](file:///C:/Users/steak/AppData/Local/Temp/msohtmlclip1/01/clip_image008.png)![1718637680740](image/README/1718637680740.png)

图3 ELAN结构

YOLOv7网络模型的颈部网络主要由SPPCSPC（Spatial Pyramid Pooling Cross Stage
Partial Connections）、CBS、ELAN、UP模块以及MP-C3模块构成。

SPPCSPC模块图4所示，SPPCSPC采用了SPP结构，通过在一系列CBS之后引入并行的多次MaxPool操作，并进行多分支的连接(cat操作)。使用MaxPool进行降维可以减少参数量，节省计算成本，加快候选框的生成速度，也可以去除特征信息的冗余。多分支的连接使特征提取更充分的同时避免了图像失真。SPPCSPC模块的使用提高了网络的泛化性，增强了网络的特征表达能力。

![](file:///C:/Users/steak/AppData/Local/Temp/msohtmlclip1/01/clip_image010.png)![1718637685504](image/README/1718637685504.png)

图4 SPPCSPC结构

UP模块图5所示，由CBS和Upsample组成，是YOLOv7中的上采样结构，目的是将提取到的Feature Map进行放大，从而以更高的分辨率进行显示图像，得到更多特征信息。

![](file:///C:/Users/steak/AppData/Local/Temp/msohtmlclip1/01/clip_image012.png)![1718637690165](image/README/1718637690165.png)

图5 UP结构

在检测头部分，YOLOv7使用的IDetect检测头有三种不同尺寸，用于检测大、中、小型的目标。随着网络的加深，特征图的尺寸逐渐减小，尺度大的检测层可以通过检测头检测出更小的目标，尺度小的检测层可以通过检测头检测出更大的目标。

## 多尺度特征融合网络

在图像处理任务中，不同尺度的特征信息对于理解和解释图像内容非常重要。较低的尺度通常包含细节和局部信息，而较高的尺度则更多地关注全局语义和上下文信息。多尺度特征融合网络是一种用于深度学习模型的技术，通过融合不同尺度的特征信息，以提高模型对多尺度数据的处理能力和表达能力，在计算机视觉任务中起到重要作用，帮助模型更好地理解图像内容。

YOLOv7模型的颈部网络采用了FPN+PAN结构。其核心思想是利用高层特征图的强语义信息进行物体分类，而底层特征图则具有更强的位置信息，有助于准确的物体定位。尽管FPN结构在提高预测特征图的语义信息方面有着不错的效果，但在理论上FPN结构会带来一定的位置信息损失。因此可以通过建立自下而上的连接通路，将位置信息传递到预测特征图中，使其同时具备高度的语义信息和位置信息，从而提高目标检测的准确性。同时PAN结构设计相对简单，并且缺乏原始信息的参与学习，容易出现训练学习偏差的情况，从而影响检测精度。

NAS-FPN[3]结构采用了多尺度特征融合，是一种基于神经架构搜索(NAS) 的特征金字塔网络。通过在大规模的搜索空间中尝试不同的网络结构组合，以找到最优的连接模式和特征融合方式。这样可以自动发现适合多尺度目标检测任务的网络结构，并提升模型的性能和效果。但是搜索得到的网络结构通常较为复杂和难以解释，限制了研究人员对网络的进一步优化和调整的能力，并且使用NAS技术需要使用大量计算资源，代价相对较高。而BiFPN对FPN+PAN结构进行了修改，在不增加计算量的前提下融合更多特征，而且引入带权重的特征融合机制来强化重要特征的表达，所以本文采用BiFPN加权双向金字塔结构替换原有的FPN+PAN结构。

## 基于BiFPN的模型改进

BiFPN结构通过通过引入双向连接，它能够在不同层级上进行特征的上下文信息传递和融合，它可以在特征金字塔的不同层级间进行信息传递和交互，使得不同尺度的特征能够相互补充和增强，网络可以同时利用底层特征的位置信息和高层特征的语义信息。这样有助于处理多尺度目标的检测问题，可以提高目标检测模型对不同尺度目标的感知能力和定位准确性。

BiFPN结构通过引入双向连接解决细节信息的丢失问题，使其可以在融合过程中保留更多的细节信息，避免特征的过度模糊和失真，从而提高目标检测模型对细节特征的感知能力。

BiFPN结构如图6所示，在FPN+PAN结构的基础上进行了改进。通过删除那些只有一条输入边的节点来优化贡献低的网络连接。同时，BiFPN在同层网络间引入了额外的边，以增加路径之间的信息流动。相比于FPN+PAN结构路径的简单连接方式，BiFPN多次重复双向路径结构使得其能够更充分地利用不同路径中的特征信息，并实现更高级别的特征融合。通过多次重复相同的层，逐渐提升特征的感知范围和语义表达能力，在目标检测任务中能够更好地融合多尺度特征，增强模型对目标的感知和定位能力，提高算法检测精度。

![1718637696445](image/README/1718637696445.png)

图6 BiFPN结构

在BiFPN中ReLU激活前引入了带权特征融合，抛弃了传统的特征融合简单的特征图叠加表示，比如使用concat(concatenate)或者shortcut(pointwise addition)连接，而不对同时加进来的特征图进行区分。权重参数能够学习不同输入特征的重要性，这样可以使网络更加关注重要的特征信息，提升模型对关键特征的感知能力。本文在YOLOv7中将颈部特征融合网络部分引入BiFPN网络，具体架构如图7所示。根据融合输入分支的数量调整实际BiFPN中的网络节点的连接关系，这样可以适应不同的输入情况，灵活地处理不同尺度和分辨率的特征融合，从而改善目标检测的准确性和鲁棒性。

![1718637873408](image/README/1718637873408.png)

图7 YOLOv7-BiFPN架构

# 5. 实验及分析：

## 实验环境及参数设置

本次实验使用Pytorch深度学习框架。PyTorch是Torch的Python版本，为开发者提供了一种直观、灵活且强大的工具，能够轻松地构建、训练和部署各种深度学习模型，具有易用性和灵活性。

本研究所选用的软硬件环境如表1所示。本文使用改进后的YOLOv7算法进行训练和测试，迭代次数(Epoch)设置为300，一次训练中所取得样本的数量(Batch Size)设置为16，图像大小为640×640，使用随机梯度下降(Stochastic Gradient Descent, SGD)进行训练，根据模型大小设置初始学习率为0.01，周期学习率为0.1，SGD动量为0.937。

表1 实验软硬件环境

| 参数         | 配置          |
| ------------ | ------------- |
| 系统环境     | Windows10     |
| CPU          | R9-3900X      |
| GPU          | RTX4090       |
| 内存         | 64G           |
| 深度学习框架 | PyTorch 2.3.0 |
| 编程语言     | Python 3.8.19 |
| 加速环境     | CUDA12.1      |

## 评价指标

本实验主要的评价指标为七项：参数量Params(Parameters)、浮点运算量(Floating Point Operations, FLOPs)精确率P(Precision)、召回率R(Recall)、F1-score(基于精确率和召回率的综合指标)、均值平均精度(mean Average Precision, mAP)、FPS(Frames Per Second)。

## 数据集介绍

在训练过程中，使用多个数据集可以获得更多、更丰富的样本数据。通过在多个数据集上进行训练，模型可以学习到更广泛的目标类别、多样化的目标外观和场景，有助于改善模型的泛化能力和检测准确率。为了提升目标检测模型的性能、增强模型的鲁棒性，并解决数据不平衡等问题，使得模型更适用于各种实际场景。本次实验数据集由MS COCO[4]、PASCAL VOC、INRIA、Caltech、Wider Person[5]数据集组成。

将各个数据集转换为YOLO标注格式后，按8：5：5：2：1的比例随机选取PASCAL VOC、MS COCO、Caltech、Wider Person、INRIA训练数据集18000张图片作为训练集，再从各自验证数据集中总计2000张图片作为总验证集，进行文件移动和标注文件的创建，其中训练集与验证集比例为9：1。

## 实验结果

本次实验训练过程图如图8所示，八个图像分别代表训练集边界框损失(Box)、训练集置信度损失(Objectness)、精确率(Precision)、召回率(Recall)、验证集边界框损失(val Box)、验证集置信度损失(val Objectness)、均值平均精度mAP(IoU=0.5)和均值平均精度mAP(IoU=0.5:0.95)。

![](file:///C:/Users/steak/AppData/Local/Temp/msohtmlclip1/01/clip_image018.png)![1718637118788](image/README/1718637118788.png)

图8 模型训练过程图

## 消融实验

本次实验将BiFPN模块引入YOLOv7并与原生YOLOv7进行对比，在MS COCO 2017、PASCAL VOC 2007、PASCAL VOC 2012、INRIA、Caltech、Wider Person数据集上的测试结果如表2所示。

表2 消融实验结果

| Model        | DataSet      | P     | R     | F1    | mAP 0.5 | mAP 0.5:0.95 | FPS  |
| ------------ | ------------ | ----- | ----- | ----- | ------- | ------------- | ---- |
| YOLOv7       | COCO 2017    | 79.6% | 71.5% | 0.754 | 79.2    | 53.5          | 74.1 |
| YOLOv7       | VOC 2007     | 89.1% | 91.9% | 0.904 | 95.9    | 78.4          | 82.0 |
| YOLOv7       | VOC 2012     | 89.6% | 89.1% | 0.894 | 94.9    | 80.9          | 82.0 |
| YOLOv7       | INRIA        | 96.8% | 91.3% | 0.940 | 97.7    | 75.0          | 52.9 |
| YOLOv7       | Caltech      | 95.4% | 88.3% | 0.917 | 94.3    | 70.8          | 87.7 |
| YOLOv7       | Wider Person | 70.0% | 56.8% | 0.627 | 60.5    | 23.3          | 64.1 |
| YOLOv7-BiFPN | COCO 2017    | 81.4% | 71.1% | 0.759 | 79.4    | 53.8          | 76.3 |
| YOLOv7-BiFPN | VOC 2007     | 91.5% | 89.6% | 0.905 | 96.1    | 78.6          | 82.6 |
| YOLOv7-BiFPN | VOC 2012     | 90.6% | 89.2% | 0.899 | 95.3    | 81.4          | 82.6 |
| YOLOv7-BiFPN | INRIA        | 92.2% | 96.2% | 0.942 | 98.2    | 76.4          | 53.8 |
| YOLOv7-BiFPN | Caltech      | 95.3% | 88.8% | 0.920 | 94.3    | 71.2          | 88.5 |
| YOLOv7-BiFPN | Wider Person | 72.8% | 56.2% | 0.634 | 61.6    | 23.7          | 64.9 |

从表中可以看出在YOLOv7引入BiFPN模块后大部分指标出现了不同程度的提升，与YOLOv7原始模型进行比较，精确率平均提升了0.6%，召回率平均提升了0.4%，F1-score平均提升了0.004，mAP@0.5平均提升了0.4，mAP@0.5:0.95平均提升了0.5，检测速度平均提升了1.4%。

通过消融实验，可以证明对YOLOv7引入BiFPN模块算法改进可以有效提升目标检测的精度和性能。

# 参考文献

[1] Tan M, Pang R, Le Q V. Efficientdet: Scalable and efficient
object detection[C]. Proceedings of the IEEE/CVF conference on computer vision
and pattern recognition. 2020: 10781-10790.

[2] Wang C Y, Bochkovskiy A, Liao H Y M. YOLOv7: Trainable
bag-of-freebies sets new state-of-the-art for real-time object detectors[J].
arXiv preprint, 2022, arXiv:2207.02696.

[3] Ghiasi G , Lin T Y , Le Q V .NAS-FPN: Learning Scalable
Feature Pyramid Architecture for Object Detection[J].IEEE,
2019.DOI:10.1109/CVPR.2019.00720.

[4] Lin T Y, Maire M, Belongie S, et al. Microsoft coco: Common
objects in context[C]//Computer Vision–ECCV 2014: 13th European Conference,
Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13. Springer
International Publishing, 2014: 740-755.

[5] Zhang S, Xie Y, Wan J, et al. Widerperson: A diverse dataset
for dense pedestrian detection in the wild[J]. IEEE Transactions on Multimedia,
2019, 22(2): 380-393.
