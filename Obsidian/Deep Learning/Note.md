格式化当前时间做文件名

```

import datetime

cur_time = datetime.datetime.now()

date_str = cur_time.strftime("%Y-%m-%d-%H-%M")

```


`torch.backends.cudnn.benchmark = True`:

让内置的 cuDNN 的 auto-tuner 自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。

适用条件：网络的输入数据维度或类型上变化不大

  

DataLoader：

+ batch_size ( int , optional ) – 每批要加载多少样本（默认值：1）

+ shuffle ( bool , optional ) – 设置为True在每个 epoch 重新洗牌数据（默认值：）False。

+ num_workers ( int , optional ) – 用于数据加载的子进程数。0表示数据将在主进程中加载​​。（默认0：）

+ pin_memory ( bool , optional ) – 如果True是，数据加载器将在返回之前将张量复制到 CUDA 固定内存中。

+ drop_last ( bool , optional ) –True如果数据集大小不能被批次大小整除，则设置为丢弃最后一个不完整的批次。如果False数据集的大小不能被批大小整除，那么最后一批将更小。（默认False）

  

坐标回归的注意事项：

1. 对坐标进行归一化操作，及 u/w, v/h

  
  

## Pytorch中常用的loss
1. L1 loss
	功能: 计算output和target之差的绝对值
	数学表达式: 
	$$ \ell(x, y) = L = \{l_1, l_2, ..., l_N\}^T, \: l_n = |x_n - y_n | $$
	```Python
	CLASS torch.nn.L1Loss 对应 torch.nn.functional.l1_loss()
	成员函数：
	- __init__(size_average=None, reduce=None, reduction='mean')-> None
	Parameters：
		size_average(bool, optional): 已弃用。
		reduce(bool, optional): 已弃用。
		reduction (string, optional): none | mean | sum
	- forward(self, input: Tensor, target: Tensor) -> Tensor
	```
	代码示例:
	```Python
	loss = nn.L1Loss()
	input = torch.randn(3, 5, requires_grad=True)
	target = torch.randn(3, 5)
	output = loss(input, target)
	output.backward()
	```
2. MSE loss (L2 loss)
	功能: 计算output和target之差的平方
	数学表达式:
	$$ \ell(x, y) = L = \{l_1, l_2, ..., l_N\}^T, \: l_n = (x_n - y_n)^2 $$
	```Python
	CLASS torch.nn.MSELoss 对应 torch.nn.functional.mse_loss()
	成员函数：
	- __init__(size_average=None, reduce=None, reduction='mean')-> None
	Parameters：
		size_average(bool, optional): 已弃用。
		reduce(bool, optional): 已弃用。
		reduction (string, optional): mean | sum | none
	- forward(self, input: Tensor, target: Tensor) -> Tensor
	```
	代码示例:
	```Python
	loss = nn.MSELoss()
	input = torch.randn(3, 5, requires_grad=True)
	target = torch.randn(3, 5)
	output = loss(input, target)
	output.backward()
	```
3. CrossEntropyLoss(交叉熵损失)
	功能：输出经过softmax激活函数后，再求其与target的交叉熵损失。
	数学表达式：
	$$ \ell(x, y) = L = \{ l_1, l_2, ..., l_N \}^T, \: l_n = -\omega_{y_n} \cdot log \frac{exp(x_{n,y_n})}{\sum_{c=1}^C\exp(x_n,c)} \cdot 1 (y_n \neq ignore\_index)$$
## torch.nn.xxx 和 torch.nn.functional.xxx 的区别
相同之处：
nn.xxx和nn.functional.xxx的功能是相同的，例如nn.Conv2d和nn.functional.conv2d 都是进行卷积。且两者的运行效率也是近乎相同。
不同之处：
1. nn.xxx 是一个类，是对nn.functional.xxx的类封装，且nn.xxx都继承于nn.Module，这导致nn.xxx除了具有nn.functional.xxx功能之外，内部附带了nn.Module相关的属性和方法，例如train(), eval(),load_state_dict(), state_dict()等；而 nn.functional.xxx 是一个函数接口
2. 调用方式不同
	nn.xxx 需要先实例化并传入超参数，然后以函数调用的方式调用实例化的对象并传入输入数据。
	示例代码：
	```Python
	inputs = torch.rand(64, 3, 244, 244)
	conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
	out = conv(inputs)
	```
	nn.functional.xxx需要同时传入输入数据和weight, bias等其他参数
	示例代码：
	```Python
	weight = torch.rand(64,3,3,3)
	bias = torch.rand(64)
	out = nn.functional.conv2d(inputs, weight, bias, padding=1)
	```
1. nn.xxx能与nn.Sequential结合使用，而nn.functional.xxx无法与nn.Sequential结合使用
2. nn.xxx不需要你自己定义和管理weight；而nn.functional.xxx需要你自己定义weight，每次调用的时候都需要手动传入weight, 不利于代码复用。
例如：
```Python

# 使用 nn.xxx 定义网络结构

class CNN(nn.Moudle)

def __init__(self):

super(CNN, self).__init__()

self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5,padding=0)

self.relu1 = nn.ReLU()

self.maxpool1 = nn.MaxPool2d(kernel_size=2)

self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=0)

self.relu2 = nn.ReLU()

self.maxpool2 = nn.MaxPool2d(kernel_size=2)

self.linear1 = nn.Linear(4 * 4 * 32, 10)

def forward(self, x):

x = x.view(x.size(0), -1)

out = self.maxpool1(self.relu1(self.cnn1(x)))

out = self.maxpool2(self.relu2(self.cnn2(out)))

out = self.linear1(out.view(x.size(0), -1))

return out

  

# 使用 nn.functional.xxx 定义网络结构

class CNN(nn.Module):

def __init__(self):

super(CNN, self).__init__()

self.cnn1_weight = nn.Parameter(torch.rand(16, 1, 5, 5))

self.bias1_weight = nn.Parameter(torch.rand(16))

self.cnn2_weight = nn.Parameter(torch.rand(32, 16, 5, 5))

self.bias2_weight = nn.Parameter(torch.rand(32))

self.linear1_weight = nn.Parameter(torch.rand(4 * 4 * 32, 10))

self.bias3_weight = nn.Parameter(torch.rand(10))

def forward(self, x):

x = x.view(x.size(0), -1)

out = F.conv2d(x, self.cnn1_weight, self.bias1_weight)

out = F.conv2d(x, self.cnn2_weight, self.bias2_weight)

out = F.linear(x, self.linear1_weight, self.bias3_weight)

return out

```

  

使用建议：

1. 具有学习参数的（例如，conv2d, linear, batch_norm)采用nn.xxx方式

2. 没有学习参数的（例如，maxpool, loss func, activation func）等根据个人选择使用nn.functional.xxx或者nn.xxx方式

3. 关于dropout，推荐使用nn.xxx方式，因为一般情况下只有训练阶段才进行dropout，在eval阶段都不会进行dropout。使用nn.Xxx方式定义dropout，在调用model.eval()之后，model中所有的dropout layer都关闭，但以nn.function.dropout方式定义dropout，在调用model.eval()之后并不能关闭dropout。


  

参考：

https://www.jianshu.com/p/5ead65699a70

  

## reduction各项值的作用：

默认值为mean

1. none: 对应每一个元素的loss, 即loss的维度与output和target的维度一致

2. sum: 所有元素的loss和

3. mean：所有元素loss的均值

示例代码:

```Python

import torch

import torch.nn as nn

loss1 = nn.MSELoss(reduction="sum")

loss2 = nn.MSELoss(reduction="mean")

loss3 = nn.MSELoss(reduction="none")

input = torch.randn(2, 3, 5)

target = torch.randn(2, 3, 5)

# print("input: ", input)

# print("target: ", target)

print("sum: ", loss1(input, target))

print("mean: ", loss2(input, target))

print("none: \n", loss3(input, target))

  

'''

>>> 输出结果:

sum: tensor(70.3890)

mean: tensor(2.3463)

none:

tensor([[[5.2777e-02, 8.1631e-03, 1.0408e-01, 2.3031e+00, 3.1570e-02],

[5.1508e-01, 3.6770e-01, 8.5001e+00, 1.7985e+00, 2.2736e-01],

[4.4870e+00, 5.8673e+00, 1.2003e+01, 7.1783e+00, 6.6738e-02]],

  

[[3.6311e+00, 2.8997e-02, 1.4072e-01, 5.1552e+00, 4.0164e+00],

[5.4172e-02, 1.2015e+00, 1.5712e+00, 2.9597e+00, 1.9255e-02],

[1.5343e-01, 7.2784e+00, 2.1082e-02, 5.9351e-01, 5.3466e-02]]])

'''

```

## 理解训练代码中optimizer.zero_grad(), loss.backward(), optimizer.step()的作用:
optimizer.zero_grad(): 将梯度归零
loss.backward()：反向传播计算得到每个参数的梯度值
optimizer.step()：通过梯度下降执行一步参数更新
一般训练代码的流程如下：
```Python

model = MyModel()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

for epoch in range(epochs):

for i, (inputs, labels) in enumerate(train_loader):

output= model(inputs)

loss = criterion(output, labels)

# compute gradient and do SGD step

optimizer.zero_grad()

loss.backward()

optimizer.step()

```

该训练代码中，每一个batch需要进行一次梯度计算以及一次参数更新。而pytorch中的backward()函数的计算时，当网络参量进行反馈时，梯度是被积累的而不是被替换，因此在每一个batch需要optimizer.zero_grad()。若一个epoch进行一次optimizer.zero_grad()，则相当与batch size为整个数据集的大小。

  
  

## heatmap

高斯核表达式：

$$G(x, y) = \frac{1}{2 \cdot \pi \cdot \sigma ^2}exp(-\frac{[(x-x_k)^2 + (y-y_k)^2}{2 \cdot \sigma ^2})$$

$(x_k, y_k)$为第k个keypoint的ground true缩放到heatmap中的位置，$\sigma$为高斯核的大小。即高斯核是一个以$(x_k, y_k)$为圆心，$\sigma /2$为半径的圆形区域

  

示例代码：

```Python
def gauss_kernel(sigma, kernel_sz = None):
    '''
    @func
        生成高斯核
    @param 
        sigma: 方差
        kernel_sz: 高斯核大小
    '''
    if kernel_sz is None:
        stride = 3*sigma
        kernel_sz = 2*stride+1
    kernel = np.zeros((kernel_sz, kernel_sz))
    center = kernel_sz // 2
    for i in range(kernel_sz):
        for j in range(kernel_sz):
            x, y = i - center, j - center
            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.max()
    return kernel
    

def gauss_heatmap(width, height, cx, cy, sigma, kernel_sz = None):
    '''
    @func
        生成heatmap
    @param 
        width: heatmap 宽度
        height: heatmap 长度
        cx, cy: 高斯核中心点坐标
        sigma: 方差
        kernel_sz: 高斯核大小
    '''
    if kernel_sz is None:
        stride = 3*sigma
        kernel_sz = 2*stride+1
    else:
        stride = kernel_sz// 2
    heatmap_padding = np.zeros((height + stride * 2, width + stride * 2))
    kernel = gauss_kernel(sigma, kernel_sz)
    heatmap_padding[cy:cy+2*stride+1, cx:cx+2*stride+1] = kernel
    heatmap = heatmap_padding[stride:stride+height, stride:stride+width]

    return heatmap


def all_kps_gauss_heatmap(width, height, kps, sigma, kernel_sz = None):
    if kernel_sz is None:
        stride = 3*sigma
        kernel_sz = 2*stride+1
    else:
        stride = kernel_sz// 2
    heatmap_padding = np.zeros((height + stride * 2, width + stride * 2))
    kernel = gauss_kernel(sigma, kernel_sz)
    for i in range(kps.shape[0]):
        cx, cy = kps[i, 0], kps[i, 1]
        heatmap_padding[cy:cy+2*stride+1, cx:cx+2*stride+1] = kernel
    
    heatmap = heatmap_padding[stride:stride+height, stride:stride+width]
    return heatmap
```

  
  
  

[ ] Adam优化器

  

[ ] F.mse_loss

  

[ ] MultiStepLR