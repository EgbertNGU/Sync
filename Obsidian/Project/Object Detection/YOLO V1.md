## 论文笔记

论文题目: You Only Look Once: Unified, Real-Time Object Detection
[论文地址](https://arxiv.org/abs/1506.02640)

### yolo的优缺点
YOLO makes more localization errors but is less likely to predict false positives on background(相比于SOTA，YOLO V1会出现更多的定位错误，但会少很多将背景当作目标的误检情况出现)

### 基本思路
1. 将输入图片划分为$S \times S$ 的网格(grid)，如果物体的中心落在某个网格中，则该网格负责检测物体
2. 每一个网格单元预测$B$个bounding box信息和以及这$B$个bounding box对应的置信度分数。bounding box的信息表示为：$x$，$y$，$w$和$h$。$(x，y)$表示bounding box的中心相对于负责该物体预测的grid cell左上点的偏移，$w$和$h$是归一化之后的宽高，即bounding box的宽高分别除以图像的宽高。$x$，$y$，$w$，$h$的范围均在0-1之间。置信度通过预测框与ground truth的IOU表示，即$Pr(\textrm{Object})*IOU_{pred}^{truth}$($Pr(\textrm{Object})$的取值为0或1)，当grid cell中没有物体存在时，$Pr(\textrm{Object})$为0，置信度分数为0；当grid cell中有物体存在时，$Pr(\textrm{Object})$为1，置信度分数等于预测框与ground truth的IOU。
3. 每一个grid cell同样会预测$C$个类别物体的类别概率$\Pr(\textrm{Class}_i | \textrm{Object})$。每个grid cell只预测的一组类别概率，即每一个grid cell只能预测一个物体。对于每一个边框，每个类别的置信度分数为：$\Pr(\textrm{Class}_i | \textrm{Object}) * \Pr(\textrm{Object}) * \textrm{IOU}_{\textrm{pred}}^{\textrm{truth}} = \Pr(\textrm{Class}_i)*\textrm{IOU}_{\textrm{pred}}^{\textrm{truth}}$


### 网络设计
![[YOLO V1结构图.png]]
共24个卷积层，2个全连接层
输入为：$448 \times 448 \times 3$的图像
输出为：$7 \times 7 \times 30$的tensor。$7 \times 7$表示将图像划分为$7 \times 7$的grid cell。每一个grid cell同时预测出bounding box，每个bounding box包含$x$，$y$，$w$，$h$和置信度5个值。目标类别共20个，每个grid cell还需要预测每一个类别的类概率。因此每个grad cell共预测$2*5+10=30$个值。
![[YOLO V1输出.png]]

### 训练过程
1. 在ImageNet 1000-class竞赛数据集上预训练前20个卷积层。预训练的图像大小为$224 \times 224$
2. 在预训练的卷积层后增加4个卷积层和2个全连接层用于目标分类任务。新增层的权重随机初始化。
3. 训练样本label构造
	 + 对象的分类概率
	>对于输入图像的某个对象S，先找到其中心点的坐标，再找出该坐标落的网格G，在G对应的label中，S的概率为1， 其余对象的概率为0；在其他的网格中，S的概率为0。
	>例如下图中，自行车的中心点位置用黑色标记，落在黄色的网格中，则黄色网格30维向量对应的label中，该自行车的概率为1，其余对象的概率为0；在其他的48个网格label中，该自行车的概率为0。
	>![[YOLO v1 label1.png]]

	+ 2个bounding box的位置
		两个bounding box的label对应位置放置的内容相同，若有物体中心落在网格G中，则对应位置均为边框中心点和宽高归一后的结果；若无物体中心落在网格G中，则对应位置均为0。
	+ 2个bounding box的置信度
		两个bounding box的label对应位置放置的内容相同。若有物体中心落在网格G中，则两个2个bounding box的置信度值均设为1；若无，则设为0。

### Loss function
$$
\begin{aligned}
	&\quad \lambda_\textbf{coord}\sum_{i = 0}^{S2}\sum_{j = 0}^{B}\mathbb{1}_{ij}^{\text{obj}}\left[\left(x_i - \hat{x}_i\right)^2 +\left(y_i - \hat{y}_i\right)^2\right] \\
	&+ \lambda_\textbf{coord}\sum_{i = 0}^{S2}\sum_{j = 0}^{B}\mathbb{1}_{ij}^{\text{obj}}\left[\left(\sqrt{w_i} - \sqrt{\hat{w}_i}\right)^2 +\left(\sqrt{h_i} - \sqrt{\hat{h}_i}\right)^2\right] \\
	&+ \sum_{i = 0}^{S2}\sum_{j = 0}^{B}\mathbb{1}_{ij}^{\text{obj}}\left(C_i - \hat{C}_i\right)^2\\
	&+ \lambda_\textrm{noobj}\sum_{i = 0}^{S2}\sum_{j = 0}^{B}\mathbb{1}_{ij}^{\text{noobj}}\left(C_i - \hat{C}_i\right)^2\\
	&+ \sum_{i = 0}^{S2}\mathbb{1}_i^{\text{obj}}\sum_{c \in \textrm{classes}}\left(p_i(c) - \hat{p}_i(c)\right)^2
\end{aligned}
$$
#### 符号的含义：
1. $\mathbb{1}_{i}^{\text{obj}}$ 表示网格$i$中存在对象
2. $\mathbb{1}_{ij}^{\text{obj}}$表示网格$i$的第$j$个bounding box中存在物体
3. $\mathbb{1}_{ij}^{\text{noobj}}$表示网格$i$的第$j$个bounding box中不存在物体
#### 每个组成部分的含义
1. $\lambda_\textbf{coord}\sum_{i = 0}^{S2}\sum_{j = 0}^{B}\mathbb{1}_{ij}^{\text{obj}}\left[\left(x_i - \hat{x}_i\right)^2 +\left(y_i - \hat{y}_i\right)^2\right]$

### 总结
1. 一张图片最多可以检测出$S \times S$个对象，与grad cell的数量相同
2. 每张图片共有$S \times S \times 2$个候选的bounding box



### 参考
1. https://zhuanlan.zhihu.com/p/46691043


