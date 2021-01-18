## fastAI第四章学习笔记-3

#### The MNIST Loss Function
原文
>We already have our independent variables x—these are the images themselves. We'll concatenate them all into a single tensor, and also change them from a list of matrices (a rank-3 tensor) to a list of vectors (a rank-2 tensor). We can do this using view, which is a PyTorch method that changes the shape of a tensor without changing its contents. -1 is a special parameter to view that means "make this axis as big as necessary to fit all the data":
```
train_x = torch.cat([stacked_threes, stacked_sevens]).view(-1, 28*28)
```
```train_x``` 也就是[笔记2](fastAI学习笔记-02-production.md)中提到的**独立变量(因变量)** ,我们是将所有3和7连起来,并将每张28\*28的矩阵转成一个28\*28的向量(列表) - 联想一下C语音二维数组转成一维数组.也就是原文说的把一个矩阵(为元素)的列表(三阶)转成了一个以向量为元素的列表.下面输出```stacked_sevens```, ```stacked_threes```和```train_x```的形状可以看出,三阶转成了2阶.
```
stacked_sevens.shape
#输出 torch.Size([6265, 28, 28])
```

```
stacked_threes.shape
#输出 torch.Size([6131, 28, 28])
```

```
train_x.shape
#输出 torch.Size([12396, 784])
```

需要标记每张图片, 用```1```来标记3的所有图片, ```0```来标记图片7的所有图片：
```
train_y = tensor([1]*len(threes) + [0]*len(sevens)).unsqueeze(1)
train_x.shape,train_y.shape
#输出 (torch.Size([12396, 784]), torch.Size([12396, 1]))
```
A Dataset in PyTorch is required to return a tuple of (x,y) when indexed. Python provides a zip function which, when combined with list, provides a simple way to get this functionality:
```
dset = list(zip(train_x,train_y))
x,y = dset[0]
x.shape,y
# 输出打印 (torch.Size([784]), tensor([1]))
```

上面准备好了训练集,现在同样的方法准备验证集：
```
valid_x = torch.cat([valid_3_tens, valid_7_tens]).view(-1, 28*28)
valid_y = tensor([1]*len(valid_3_tens) + [0]*len(valid_7_tens)).unsqueeze(1)
valid_dset = list(zip(valid_x,valid_y))
```

现在要为每个像素初始化一个weight(权重)这也是七步中的initialize步：

```
def init_params(size, std=1.0): return (torch.randn(size)*std).requires_grad_()
weights = init_params((28*28,1))

weights.shape
# 这里会打印torch.Size([784, 1])，也就是说生成了一个长度为784的向量weight，稍后会用这个向量(列数为1的矩阵)做乘法.
```

线性函数,只有这一个参数还不够```y=wx+b```这样的一次函数还有一个偏移量```b```所以还需要初始化一个随机数
```
bias = init_params(1)
```
在神经网络中,式子```y=wx+b```里的```w```叫 _wights(权重)_, ```b```叫 _bias(偏移)_  
试一下计算第一张图片的预测：
```
(train_x[0]*weights.T).sum() + bias
# 这里打印输出 tensor([25.2817], grad_fn=<AddBackward0>)
```
***疑问，为什么不是每个像素都有一个```bias```，而是一个图片的像素都是一个bias***  
现在要计算所有图片的预测值,在python中,用矩阵的乘法比循环会快很多,操作符```@```表示矩阵相乘.
```
def linear1(xb): return xb@weights + bias
preds = linear1(train_x)
preds
'''
这里会打印：
tensor([[ 25.2817],
        [ 22.5130],
        [ 23.5370],
        ...,
        [-20.4941],
        [-22.4730],
        [-18.8594]], grad_fn=<AddBackward0>)
'''
```
可以看出第一元素的计算结果和前面的计算一样.

现在检查下准确性。由于我们已经用train_y标记了一张图片是3还是7,所以用train_y做判断
```
train_y.T
'''
会打印：
tensor([[1, 1, 1,  ..., 0, 0, 0]])
在定义train_y时,前几个标记的是1，后面标记的是0, .T 是转置矩阵
···
```
```
corrects = (preds>0.0).float() == train_y
correct
'''
会打印：
tensor([[ True],
        [ True],
        [ True],
        ...,
        [ True],
        [False],
        [False]])
'''
```
