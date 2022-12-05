## GPU训练
1. `.cuda()`
2.`device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")`
	`.to(device)`


## `model.train()`和`model.eval()`的用法和区别
1. `model.train()`的作用是启用 Batch Normalization 和 Dropout。如果模型中有BN层(Batch Normalization）和Dropout层，需要在训练时添加model.train()保证BN层能够用到每一批数据的均值和方差以及Dropout层随机取一部分网络连接来训练更新参数。
2. `model.eval()`的作用是不启用 Batch Normalization 和 Dropout。如果模型中有BN层(Batch Normalization）和Dropout，在测试时添加model.eval()，保证BN层能够用全部训练数据的均值和方差，即测试过程中要保证BN层的均值和方差不变以及Dropout层可以利用到所有网络连接，即不进行随机舍弃神经元。在测试和验证阶段，如果网络中有BN层和Dropout层则需要在开始时添加`model.eval()`
3. `train`的API为`train(mode=True)`，mode的取值为True和False；`eval`的API为`eval()`。
	且`model.eval()`等价于`model.train(False)`