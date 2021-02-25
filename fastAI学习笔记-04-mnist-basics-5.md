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
1. Explain how the "pixel similarity" approach to classifying digits works.
1. What is a list comprehension? Create one now that selects odd numbers from a list and doubles them.
1. What is a "rank-3 tensor"?
1. What is the difference between tensor rank and shape? How do you get the rank from the shape?
1. What are RMSE and L1 norm?
1. How can you apply a calculation on thousands of numbers at once, many thousands of times faster than a Python loop?
1. Create a 3×3 tensor or array containing the numbers from 1 to 9. Double it. Select the bottom-right four numbers.
1. What is broadcasting?
1. Are metrics generally calculated using the training set, or the validation set? Why?
1. What is SGD?
1. Why does SGD use mini-batches?
1. What are the seven steps in SGD for machine learning?
1. How do we initialize the weights in a model?
1. What is "loss"?
1. Why can't we always use a high learning rate?
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
