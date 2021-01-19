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
>在[fastAI 论坛发帖](https://forums.fast.ai/t/in-04-mnist-basics-why-bias-is-not-for-every-independent-pixel/84769),[jimmiemunyi](https://forums.fast.ai/u/jimmiemunyi) 拿了一个神经元的图解释,确实是对sum加bias, 但似乎没有解释为什么bias加在这个地方.

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
```
corrects.float().mean().itern()
# 会打印 0.4912068545818329
```

把weights 做微小调整：
```
weights[0] *= 1.0001
preds = linear1(train_x)
((preds>0.0).float() == train_y).float().mean().item()

# 会输出0.4912068545818329
```
可以看到调整了weights后,输出结果没有任何变化.因为函数的梯度是它的斜率或陡度,即函数值变化的幅度除以输入值改变的幅度：
```(y_new - y_old)/(x_new - x_old)```. 当x_new和x_old非常接近时,就可以很好的渐变.但是从输出来看,只有预测值从3变为7或由7变为3时,预测精度才会有变化。但因为weights 从x_old到x_new的变化太小,不足以导致y的变化.所以```(y_new - y_old)```几乎始终为0.换句话说,几乎所有地方的梯度都为0.
但是我们需要使用SGD改进模型,梯度都0的话就不能改进这个模型了.  
为了解决这个问题,我们要用一个损失函数,当我们的权重得出更好的预测时,它会返回更好的loss .更好的预测，即：
判断为3的可能性越大,分数越高,判断为7的可能性越大(即意味着判断为3的可能性越小)则分数越低.  

损失函数输入的不是图像本身,而是来自模型的预测值.用一个参数prds表示介于0和1之间的值,每个值表示预测为3的可能性.
损失函数的目的是衡量预测值和真实值(即target也称为label),他表明图像的实际情况是否真的为3.   
假设我们有三个图像,分别是3,7,3 并且假设我们模型预测第一为3的可信度是0.9，第二个不为3(为7)的可信度为0.4, 第三个为不为3(为7)的可信度为0.2 。其中前两个判断真实值一致,而第三个判断错了.
```
trgts = tensor([1, 0, 1])
prds  = tensor([0.9, 0.4, 0.2])
```
先尝试一个损失函数loss function衡量预测值和真实值之间的差距
```
def mnist_loss(predictions, targets):
    return torch.where(targets==1, 1-predictions, predictions).mean()
```
其中``` torch.where(targets==1, 1-predictions, predictions).mean()```类似于C语言语句```targets == 1 ? 1-predictions : predictions```
也就是说如果预测为3(即trgts为1)，则取1-predictions表示"错误度". 如果预测不为3，则直接取predictions表示"错误度"，然后取"错误度"的平均值表示损失loss.
```
torch.where(trgts==1, 1-prds, prds)
# 输出tensor([0.1000, 0.4000, 0.8000])
```
从上面的输出值可以看出这个值小,"预测的正确性"越大.然后对这一组取平均值,就可以很好的反应“预测好不好”
```
mnist_loss(prds,trgts)
# 输出tensor(0.4333)
```
如果我们把prds中的0.2变为0.8,

### Sigmoid
