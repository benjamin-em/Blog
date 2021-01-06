## fastAI第四章学习笔记第二部分：导数和梯度

### PyTorch 计算导数

PyTorch提供非常简单的计算某一点导数的方法
```
def f(x): return 2*(x**3) + 3*(x**2) + 4*x + 1
```
假设要对```(1.5, f(1.5))```这个点求导.定义一个张量,并使用```requires_grad_()```标记它需要求导,注意带下划线表示是一个原位操作,即计算结果会覆盖原有值.
```
xt = tensor(1.5).requires_grad_()
```
将xt代入函数,计算出函数值,yt
```
yt = f(xt)
```
然后求```yt```的导数值,```backward()```就是求导的函数,这个命名是取自 _backpropagation_ 术语反向传播
```
yt.backward()
xt.grad
#这里会打印tensor(26.5000)
```

还可以将一组向量作为参数传给函数,在函数内对向量进行运算,这里求和.
```
def f_sum(x) : return f(x).sum()

xt = tensor([3.,4.,10.]).requires_grad_()

yt = f_sum(xt)
yt
#这里会打印tensor(2628., grad_fn=<SumBackward0>)

yt.backward()
xt.grad
#这里会打印 tensor([ 76., 124., 664.]) , 即向量的导数值.
```

### SGD应用举例
想象过山车爬坡又下来的情形：
上坡速度越来越慢,到顶后下坡速度又越来越快。20秒,每秒测一次速度.
```
time = torch.arange(0,20).float();

speed = torch.randn(20)*3 + 0.75*(time-9.5)**2 + 1  #这里加了一个随机速度.

plt.scatter(time,speed); #绘制时间/速度坐标图像.
```
我们**猜想**这个函数曲线是一个二次函数```a*(time**2)+(b*time)+c```

为了区分**输入 - input**和**参数 - parameter**, 将**输入**和**参数**分开传进一个函数：
```
def f(t, params):
    a,b,c = params
    return a*(t**2) + (b*t) + c
```
为了找到**最佳**二次函数,只需要找到最合适的参数```a,b,c```. 用均方差来衡量是否**最佳**
```
def mse(preds, targets): return ((preds-targets)**2).mean().sqrt()
```
这里```preds```将会传入预测的值,也就是函数```f(t, params)```的返回值. ```targets```将会传入实际测量的速度.

现在用7步：
#### 第一步：初始化参数
最佳初始一组参数就可以,用```requires_grad_```告诉PyTorch需要跟踪他们的梯度.
```
params = torch.randn(3).requires_grad_()

orig_params = params.clone()
```
#### 第二步：计算预测值：
```
preds = f(time, params)
# 这里time就是前面定义的0-19秒,每秒的时刻.
```
这个定义一个函数画出图形：
```
def show_preds(preds, ax=None):
    if ax is None: ax=plt.subplots()[1]
    ax.scatter(time, speed)
    ax.scatter(time, to_np(preds), color='red')
    ax.set_ylim(-20,100)

show_preds(preds)
```
![show_pred](img/show_preds.jpg)
#### 第三步：计算loss：
```
# mse() 为 ((preds-targets)**2).mean().sqrt()
# preds是传入不同的时刻t, a*(t**2) + (b*t) + c的计算的speed值, 
# targes,在不同的t测量出来的speed. 

loss = mse(preds, speed)
loss
#这里打印 tensor(25.2016, grad_fn=<SqrtBackward>)
```
我们目标是减小均方差,**所以需要计算梯度**
#### 第四步：计算梯度
计算梯度, 也就是说计算一组一个近似值,指导**参数**如何变化.
```
loss.backward()
params.grad
tensor([-0.0052,  0.1076, -0.3424])
```
需要注意的是这里是对params求梯度也就是对下面的```a,b,c```求梯度.
 ```
 (a*t**2 + b*t + c - speed)**2).mean().sqrt()
 ```

