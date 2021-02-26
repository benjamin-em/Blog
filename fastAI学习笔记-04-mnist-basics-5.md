## Jargon Recap and Questionnaire

### 术语
神经网络包含很多数字，但它们只有两类：
- 激活值(Activations):: 计算出来的值(包括线性和非线性层)
- 参数(parameters):: 那些随机初始化和优化过的数字(也就是,定义模型的那些数字)  
它们不是抽象概念,而是你的模型中具体的数字. 成为一名优秀的深度学习实践者的一部分,是习惯了实际查看激活和参数,绘制它们并测试它们是否行为正确的想法。
所有的激活值和参数都是由 _tensors_ 表示的。有些只是规则形状的数组,例如矩阵。 矩阵有行和列; 我们称这些为axes或dimensions。
这是一些特别的tensors:

- Rank zero(0阶): 标量
- Rank one(1阶): 矢量
- Rank two(2阶): 矩阵

一个神经网络可以有很多层,每一层要么是线性的,要么是非线性的. 我们一般交替地把它们放在一个神经网络里。有时人们将线性层及其随后的非线性都称为单个层. 嗯, 这令人困惑.
有时，非线性被称为 _激活函数(activation function)_ .

| Term | Meaning|
|----|---|
|ReLU | Function that returns 0 for negative numbers and doesn't change positive numbers.
|Mini-batch | A small group of inputs and labels gathered together in two arrays. A gradient descent step is updated on this batch (rather than a whole epoch).
|Forward pass | Applying the model to some input and computing the predictions.
|Loss | A value that represents how well (or badly) our model is doing.
|Gradient | The derivative of the loss with respect to some parameter of the model.
|Backward pass | Computing the gradients of the loss with respect to all model parameters.
|Gradient descent | Taking a step in the directions opposite to the gradients to make the model parameters a little bit better.
|Learning rate | The size of the step we take when applying SGD to update the parameters of the model.

### 问题
1. How is a grayscale image represented on a computer? How about a color image?  
  以0-255为范围,从白(0)到黑(255)之间的数字灰度显示一幅黑白(灰度)像素.彩色图像是RGB三种颜色按一定比例混合产生一个颜色像素.
1. How are the files and folders in the `MNIST_SAMPLE` dataset structured? Why?  
  它将训练集,验证集文件按目录分开,并在下一级目录中,按数字名来存放对应的数字如"train/7"是训练集下数字7的目录.这样以目录的形式可以很方便地区分验证集,测试集.
1. Explain how the "pixel similarity" approach to classifying digits works.  
  将验证集中的数字图像堆叠并将堆叠在同一位置的像素取平均值，得到“理想”的数字图像,将待检测的数字图像与“理想”图像比较检测其差距大小.
1. What is a list comprehension? Create one now that selects odd numbers from a list and doubles them.  
  列表推导式, ```[x*2 for x in range(1,11) if x%2 != 0]```
1. What is a "rank-3 tensor"?  
  rank-3 tensor 是一系列rank-2 tensor,也就一系列矩阵.
1. What is the difference between tensor rank and shape? How do you get the rank from the shape?  
  shape描述了每个轴的长度,也就是每个阶的大小; rank指shape的长度, 也就说tensor有几阶.
1. What are RMSE and L1 norm?  
  RMSE指root mean squared error，也就是差的平方取平均值再开放; L1 范式指对差的绝对值取平均值
1. How can you apply a calculation on thousands of numbers at once, many thousands of times faster than a Python loop?  
  利用并行计算能力,使用广播(broadcasting)进行计算.
1. Create a 3×3 tensor or array containing the numbers from 1 to 9. Double it. Select the bottom-right four numbers.  
  代码如下```T = tensor(range(1,10)).view(-1,3)
  T1 = T*2
  Tbr4 = T1[:2, :2]```
1. What is broadcasting?  
  在张量和数字间进行运算时,把数字假设为相同形状的张量,再在两个张量间进行并行计算,它不会耗费更多额外内存.
1. Are metrics generally calculated using the training set, or the validation set? Why?  
  metrics一般是用验证集计算的,因为这样可以避免无意间过拟合,因为模型可能已经记住训练集的数据,使用训练集的话会导致预测不准.
  metrics只是作为人为判断模型好坏的量化标准. 但它可能不具有平滑,无跳变的特征,无法求导来反应微小变化,所以无法对参数进行优化,不宜作为
1. What is SGD?  
  SGD(stochastic gradient descent)随机梯度递减,
1. Why does SGD use mini-batches?  
  Calculating it for the whole dataset would take a very long time. Calculating it for a single item would not use much information, so it would result in a very imprecise and unstable gradient. That is, you'd be going to the trouble of updating the weights, but taking into account only how that would improve the model's performance on that single item. So instead we take a compromise between the two: we calculate the average loss for a few data items at a time. Another good reason for using mini-batches rather than calculating the gradient on individual data items is that, in practice, we nearly always do our training on an accelerator such as a GPU. These accelerators only perform well if they have lots of work to do at a time, so it's helpful if we can give them lots of data items to work on. Using mini-batches is one of the best ways to do this. However, if you give them too much data to work on at once, they run out of memory—making GPUs happy is also tricky!
1. What are the seven steps in SGD for machine learning?
   1.Initialize the parameters;
   2.Calculate the predictions
   3.Calculate the loss
   4.Calculate the gradients
   5.Step the weights - update the parameters based on the gradients we just calculated
   6.Repeat the process from Step 2 to Step 4
   7.Stop if we think the model is good enough
1. How do we initialize the weights in a model?
   we initialize the parameters to random values, and tell PyTorch that we want to track their gradients, using requires_grad_:   
1. What is "loss"?   
   We need some function that will return a number that is small if the performance of the model is good (the standard approach is to treat a small loss as good, and a large loss as bad, although this is just a convention).
1. Why can't we always use a high learning rate?
   If the learning rate is too high, it may also "bounce" around, rather than actually diverging
1. What is a "gradient"?

1. Do you need to know how to calculate gradients yourself?
1. Why can't we use accuracy as a loss function?
1. Draw the sigmoid function. What is special about its shape?
1. What is the difference between a loss function and a metric?
1. What is the function to calculate new weights using a learning rate?
1. What does the `DataLoader` class do?
1. Write pseudocode showing the basic steps taken in each epoch for SGD.
1. Create a function that, if passed two arguments `[1,2,3,4]` and `'abcd'`, returns `[(1, 'a'), (2, 'b'), (3, 'c'), (4, 'd')]`. What is special about that output data structure?
1. What does `view` do in PyTorch?
1. What are the "bias" parameters in a neural network? Why do we need them?
1. What does the `@` operator do in Python?
1. What does the `backward` method do?
1. Why do we have to zero the gradients?
1. What information do we have to pass to `Learner`?
1. Show Python or pseudocode for the basic steps of a training loop.
1. What is "ReLU"? Draw a plot of it for values from `-2` to `+2`.
1. What is an "activation function"?
1. What's the difference between `F.relu` and `nn.ReLU`?
1. The universal approximation theorem shows that any function can be approximated as closely as needed using just one nonlinearity. So why do we normally use more?
