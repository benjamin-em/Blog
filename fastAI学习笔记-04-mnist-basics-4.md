## fastAI第四章学习笔记-3
### Creating an Optimizer

首先我们可以用PyTorch中的nn.Linear模块(module)代替[前文](fastAI学习笔记-04-mnist-basics-3.md)中的linear1. _module_ 是一个从```nn.Module```类继承的子类的对象. 此类的对象的行为与标准Python函数相同，因为你可以使用括号来调用它们，并且它们将返回模型的激活值。  
```nn.Linear```的作用和我们前面自己实现的```init_params```和```linear```两个函数一起一样.它一个类里同时包含了_weihts_ 和 _bias_ . 看看是怎样替代前文的部分：
```
linear_model = nn.Linear(28*28, 1)
```
每个PyTorch 模块都知道可以训练哪些参数。 它们可通过```parameters```方法获得：
```
w,b = linear_model.parameters()
w.shape,b.shape
```
>(torch.Size([1, 784]), torch.Size([1]))  

我们可以使用此信息来创建优化器
```
class BasicOptim:
    def __init__(self,params,lr): self.params,self.lr = list(params),lr

    def step(self, *args, **kwargs):
        for p in self.params: p.data -= p.grad.data * self.lr

    def zero_grad(self, *args, **kwargs):
        for p in self.params: p.grad = None
```
我们可以通过传入模型的参数来创建优化器
```
opt = BasicOptim(linear_model.parameters(), lr)
```
我们的训练循环现在可以简化为:
```
def train_epoch(model):
    for xb,yb in dl:
        calc_grad(xb, yb, model)
        opt.step()
        opt.zero_grad()
```
我们的验证函数完全不需要改变：
```
validate_epoch(linear_model)
```
>0.4608

让我们将小的训练循环放入一个函数中，以使事情变得更简单：
```
def train_model(model, epochs):
    for i in range(epochs):
        train_epoch(model)
        print(validate_epoch(model), end=' ')
```
结果与上一节相同：
```
train_model(linear_model, 20)
```
>0.4932 0.7686 0.8555 0.9136 0.9346 0.9482 0.957 0.9634 0.9658 0.9678 0.9697 0.9717 0.9736 0.9746 0.9761 0.977 0.9775 0.9775 0.978 0.9785  

fastai提供一个```SGD```类，默认情况下，该类与```BasicOptim```起相同的作用：
```
linear_model = nn.Linear(28*28,1)
opt = SGD(linear_model.parameters(), lr)
train_model(linear_model, 20)
```
>0.4932 0.8179 0.8496 0.9141 0.9346 0.9482 0.957 0.9619 0.9658 0.9673 0.9692 0.9712 0.9741 0.9751 0.9761 0.9775 0.9775 0.978 0.9785 0.979 

fastai还提供了```Learner.fit```，我们可以使用它代替```train_model```。 要创建一个```Learner```，我们首先需要通过传递我们的训练```DataLoader```和验证```DataLoader``` 来创建一个```DataLoaders```：
```
dls = DataLoaders(dl, valid_dl)
```
要不适用一个应用(例如cnn_learner)创建一个```Learner```, 我们就需要传入所有这一章创建的元素：```DataLoaders```， 模型，优化器函数(会传入参数给它),损失函数,以及(可选的)任意用来打印指标.
```
learn = Learner(dls, nn.Linear(28*28,1), opt_func=SGD,
                loss_func=mnist_loss, metrics=batch_accuracy)
```
现在我们调用```fit```
```
learn.fit(10, lr=lr)
```
>epoch	train_loss	valid_loss	batch_accuracy	time
0	0.636709	0.503144	0.495584	00:00
1	0.429828	0.248517	0.777233	00:00
2	0.161680	0.155361	0.861629	00:00
3	0.072948	0.097722	0.917566	00:00
4	0.040128	0.073205	0.936212	00:00
5	0.027210	0.059466	0.950442	00:00
6	0.021837	0.050799	0.957802	00:00
7	0.019398	0.044980	0.964181	00:00
8	0.018122	0.040853	0.966143	00:00
9	0.017330	0.037788	0.968106	00:00

如你所见，PyTorch和fastai类没有什么神奇之处。 它们只是方便的预包装件，让你更轻松！ （它们还提供了很多额外的功能，我们将在以后的章节中使用。）
通过这些类，我们现在可以将神经网络替换为线性模型。
