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
```
>torch.Size([6265, 28, 28])

```
stacked_threes.shape
```
>torch.Size([6131, 28, 28])

```
train_x.shape
```
>torch.Size([12396, 784])

需要标记每张图片, 用```1```来标记3的所有图片, ```0```来标记图片7的所有图片：
```
train_y = tensor([1]*len(threes) + [0]*len(sevens)).unsqueeze(1)
train_x.shape,train_y.shape
```
>(torch.Size([12396, 784]), torch.Size([12396, 1]))

A Dataset in PyTorch is required to return a tuple of (x,y) when indexed. Python provides a zip function which, when combined with list, provides a simple way to get this functionality:
```
dset = list(zip(train_x,train_y))
x,y = dset[0]
x.shape,y
```
>(torch.Size([784]), tensor([1]))

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
```
>torch.Size([784, 1])  

也就是说生成了一个长度为784的向量weight，稍后会用这个向量(列数为1的矩阵)做乘法.


线性函数,只有这一个参数还不够```y=wx+b```这样的一次函数还有一个偏移量```b```所以还需要初始化一个随机数
```
bias = init_params(1)
```
在神经网络中,式子```y=wx+b```里的```w```叫 _wights(权重)_, ```b```叫 _bias(偏移)_  
试一下计算第一张图片的预测：
```
(train_x[0]*weights.T).sum() + bias
```
tensor([25.2817], grad_fn=<AddBackward0>)

***疑问，为什么不是每个像素都有一个```bias```，而是一个图片的像素都是一个bias***  
>在[fastAI 论坛发帖](https://forums.fast.ai/t/in-04-mnist-basics-why-bias-is-not-for-every-independent-pixel/84769),[jimmiemunyi](https://forums.fast.ai/u/jimmiemunyi) 拿了一个神经元的图解释,确实是对sum加bias, 但似乎没有解释为什么bias加在这个地方.

现在要计算所有图片的预测值,在python中,用矩阵的乘法比循环会快很多,操作符```@```表示矩阵相乘.
```
def linear1(xb): return xb@weights + bias
preds = linear1(train_x)
preds
```
>tensor([[ 25.2817],  
        [ 22.5130],  
        [ 23.5370],  
        ...,  
        [-20.4941],  
        [-22.4730],  
        [-18.8594]], grad_fn=<AddBackward0>)

可以看出第一元素的计算结果和前面的计算一样.

现在检查下准确性。由于我们已经用train_y标记了一张图片是3还是7,所以用train_y做判断
```
train_y.T
```
>tensor([[1, 1, 1,  ..., 0, 0, 0]])

在定义train_y时,前几个标记的是1，后面标记的是0, .T 是转置矩阵


```
corrects = (preds>0.0).float() == train_y
correct
```
>tensor([[ True],  
        [ True],  
        [ True],  
        ...,  
        [ True],  
        [False],  
        [False]])  
        
```
corrects.float().mean().itern()
```
>0.4912068545818329

把weights 做微小调整：
```
weights[0] *= 1.0001
preds = linear1(train_x)
((preds>0.0).float() == train_y).float().mean().item()
```
>0.4912068545818329

可以看到调整了weights后,输出结果没有任何变化.因为函数的梯度是它的斜率或陡度,即函数值变化的幅度除以输入值改变的幅度：
```(y_new - y_old)/(x_new - x_old)```. 当x_new和x_old非常接近时,就可以很好的渐变.但是从输出来看,只有预测值从3变为7或由7变为3时,预测精度才会有变化。但因为weights 从x_old到x_new的变化太小,不足以导致y的变化.所以```(y_new - y_old)```几乎始终为0.换句话说,几乎所有地方的梯度都为0.
但是我们需要使用SGD改进模型,梯度都0的话就不能改进这个模型了.  
为了解决这个问题,我们要用一个损失函数,当我们的权重得出更好的预测时,它会返回更好的loss .更好的预测，即：
判断为3的可能性越大,分数越高,判断为7的可能性越大(即意味着判断为3的可能性越小)则分数越低.  

损失函数输入的不是图像本身,而是来自模型的预测值.用一个参数prds表示介于0和1之间的值,每个值表示预测为3的可能性.
损失函数的目的是衡量预测值和真实值(即target也称为label),他表明图像的实际情况是否真的为3.   
原文:
> So, for instance, suppose we had three images which we knew were a 3, a 7, and a 3. And suppose our model predicted with high confidence (0.9) that the first was a 3, with slight confidence (0.4) that the second was a 7, and with fair confidence (0.2), but incorrectly, that the last was a 7.   

~~假设我们有三个图像,分别是3,7,3 并且假设我们模型预测第一为3,可信度是0.9;第二个为7,可信度为0.4;第三个为不为,可信度为0.2 。其中前两个判断真实值一致,而第三个判断错了~~  
这个confidence理解成“可信度”可能不准确。

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
```
>tensor([0.1000, 0.4000, 0.8000])

从上面的输出值可以看出这个值小,"预测的正确性"越大.然后对这一组取平均值,就可以很好的反应“预测好不好”
```
mnist_loss(prds,trgts)
```
>tensor(0.4333)

如果我们把prds中的0.2变为0.8,loss 会变小,这也就意味着得到更好的预测.
```
mnist_loss(tensor([0.9, 0.4, 0.8]),trgts)
```
>tensor(0.2333)


### Sigmoid
前面提到“用一个参数prds表示介于0和1之间的值”,怎样是prds在0和1直接？Sigmoid函数可以做到这一点.
定义如下
```
def sigmoid(x): return 1/(1+torch.exp(-x))
```
画出图形如下
```
plot_function(torch.sigmoid, title='Sigmoid', min=-4, max=4)
```
![Sigmoid](img/Sigmoid_img.jpg)  
可以看到Sigmoid是一条在0到1之间光滑且单调上升的曲线,这就很容易满足SGD中寻找梯度.
现在重新定义loss函数：
```
def mnist_loss(predictions, targets):
    predictions = predictions.sigmoid()
    return torch.where(targets==1, 1-predictions, predictions).mean()
```

我们既然有了一个总体准确性(即前面的```corrects.float().mean().itern()```或```((preds>0.0).float() == train_y).float().mean().item()```),那为什么还要定义损失函数呢？那是因为总体准确性是让人更容易判断模型好坏,而损失函数是为了机器学习.损失函数必须是具有有意义的导数的函数,不能有较大的平坦部分和较大的跳动,而且必须平滑. 这就是为什么我们要设计一个损失函数对置信度的微小变化做出响应。但这也就意味着他不能真正反映我们要实现的目标,而实际是我们实际目标和可使用梯度进行优化的功能之间的折中.损失函数会在数据集的每个项目中计算用到,然后在一个周期结束时,对所有损失值
进行平均,然后报告这个周期的总体平均值.  
另一方面,总体准确性指标是我们真正关心的数字.这在每个周期结束时会打印,这些值告诉我们模型的实际运行情况.**在判断模型性能时,我们更关心这个指标而不是损失值**.

### SGD and Mini-Batches
在有了合适的损失函数后,会根据梯度来更新weights,这个叫做优化步骤(_optinization step_)
我们可以每次针对一项数据进行计算,这样一个一个迭代计算,但是这样会很慢.也可以将所有的数据放一个巨大的矩阵中进行计算(在矩阵计算可以很好利用GPU,从而快速计算),但这样内存不够.
所以折中地,一次计算其中几项数据的平均损失.这个叫 _Mini-Batches_. 一小批中数据的个数叫做 _batch size_
PyTorch和fastAi提供一个类,可以对数据集进行随机化,并分批.
```
coll = range(15)
dl = DataLoader(coll, batch_size=5, shuffle=True)
list(dl)
```
上面```shuffle=True```表示随机化
>[tensor([ 0,  7,  4,  5, 11]),
 tensor([ 9,  3,  8, 14,  6]),
 tensor([12,  2,  1, 10, 13])]
 
 为了训练模型,我们不只是需要数据的集合,这个集合还需要包含独立变量和标记(input 和targets),这两个构成一个元组,这些元组的集合子PyTorch中叫做Dataset,举个非常简单的例子：
```
ds = L(enumerate(string.ascii_lowercase))
ds
```
>(#26) [(0, 'a'),(1, 'b'),(2, 'c'),(3, 'd'),(4, 'e'),(5, 'f'),(6, 'g'),(7, 'h'),(8, 'i'),(9, 'j')...]

把Dataset传给DataLoader会获得很多个批batches:
```
dl = DataLoader(ds, batch_size=6, shuffle=True)
list(dl)
```
>[(tensor([ 6, 14, 12, 15, 24, 11]), ('g', 'o', 'm', 'p', 'y', 'l')),   
 (tensor([ 0, 16,  2, 18, 25, 21]), ('a', 'q', 'c', 's', 'z', 'v')),  
 (tensor([ 8,  7, 19, 23,  1,  9]), ('i', 'h', 't', 'x', 'b', 'j')),  
 (tensor([ 4, 13, 10,  5,  3, 17]), ('e', 'n', 'k', 'f', 'd', 'r')),  
 (tensor([22, 20]), ('w', 'u'))] 
 
 ### Putting It All Together
