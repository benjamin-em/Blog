## FastAI 第7章学习笔记 - Training a State-of-the-Art Model

>先安装和加载必要的库  

```
!pip install -Uqq fastbook
import fastbook
fastbook.setup_book()

from fastbook import *
```

这章介绍了用于训练图像分类模型并获得最新结果的更高级技术.  介绍归一化(normalization), 一种称为Mixup的强大数据增强技术, 渐进式调整大小方法, 以及测试时数据增强技术. 首先会用到一个叫做Imagenette的ImageNet子集, 从头开始训练 - 不使用迁移学习. 它包含ImageNet中10个差异很大的类别子集, 可以在我们进行实验时进行快速训练.

### Imagenette

在fast.ai开创之初, 人们用三个主要的数据集创建和测试计算机视觉模型:

- ImageNet:: 一百三十万张一千个种类,约500像素不同尺寸的图片,这些图片花了几天训练.
- MNIST:: 50,000张28×28像素的灰度手写数字.
- CIFAR10:: 60,000 张32×32像素的不同种类彩色图片

​    这里有个问题, 较小的数据集不能有效地概括庞大的ImageNet数据集. 对ImageNet有效的的方法, 一般必须在ImageNet的基础上开发和训练. 这使得很多人相信, 只有那些拥有海量算力资源的研究员才能有效地为开发图像识别算法做贡献.

我们认为很可能不是这样的. 没有研究表明只有ImageNet 正好是合适的大小, 而不能创建其他数据集来提供有效参考. 所以我们想创建一个新的数据集, 让研究人员能用来高效率低成本的测试他们的算法, 但也将提供可能适用于整个ImageNet数据集的参考.

大约3个小时候, 创建了Imagenette. 这是从完整的ImageNet中选出10个相互差别非常大的种类. 如期望的一样, 我们能够快速而廉价地创建能够识别这些类别的分类器. 然后尝试了一些算法微调,看看它们如何影响Imagenette. 结果发现有些效果相当不错, 然后在ImageNet 上也做了测试, 然后很高兴地发现在ImageNet上这些调整也非常有效.

你拿到的数据集并不一定是你想要. 尤其不太可能是你要在其中进行开发和原型制作的数据集. 这时你应该致力于不超过几分钟的时间进行迭代尝试,也就是说能够训练模型并在几分钟之内查看其运行情况. 如果花太长时间做实验, 就该考虑裁剪数据集, 或简化模型来提高试验速度. 尝试试验越多越好.

看看Imagenette：

```
from fastai.vision.all import *
path = untar_data(RULs.IMAGENETTE)
```

首先吧dataset 搞到`DataLoaders`对象里, 并用presizing. 在之前宠物品种分类章节提到过, item_tfms是在CPU上将图片大小对齐为统一尺寸, batch_tfms是在GPU(如果启用了GPU的话)上将图片进行统一变换.

```
dblock = DataBlock(blocks=(ImageBlock(), CategoryBlock)),
                           get_items=get_image_files,
                           get_y=parent_label,
                           item_tfms=Resize(460),
                           batch_tfms=aug_transforms(size=224, min_scale = 0.75)
dls = dblock.dataloaders(path, bs=64)
```

做一次训练作为基准:

```
model = xresnet50(n_out_dls.c)
learn = Learn(dls, model, loss_func=CrossEntropyLossFlat), metrics=accuracy)
learn.fit_one_cycle(5, 3e-3)
```
>
| epoch | train_loss | valid_loss | accuracy | time  |
| ----- | ---------- | ---------- | -------- | ----- |
| 0     | 1.561049   | 4.572206   | 0.272218 | 02:45 |
| 1     | 1.202027   | 2.029665   | 0.511576 | 02:50 |
| 2     | 0.921744   | 1.096028   | 0.671770 | 02:51 |
| 3     | 0.737844   | 0.666281   | 0.786781 | 02:54 |
| 4     | 0.599707   | 0.550943   | 0.819268 | 02:54 |

这是个不错的基线, 因为没有预训练模型, 但是可以做到更好. 当一个模型是从头开始训练, 或者针对另一个和之前预训练的区别很大的数据集fine-tune(微调)时, 有一些附加的技术非常重要. 第一个就是 *归一化* (Normalizating)数据.

### Normalization

训练模型时, 如果输入数据被归一化 - 即平均值为0, 标准偏差为1, 会很有用. 但是大多数图像和计算机图形库使用的值是介于0到255之间或0到1之间的. 无论哪种情况, 数据都不会有平均值为1, 标准偏差为1的情况.

我们获取一批数据并通过平均除通道轴(即轴1)以外的所有轴来查看这些值:

```
x,y = dls.one_batch()
x.mean(dim=[0, 2, 3]), x.std(dim=[0, 2, 3])
```
>(TensorImage([0.4688, 0.4575, 0.4356], device='cuda:0'),  
 TensorImage([0.2882, 0.2813, 0.2926], device='cuda:0'))

看来, 平均值和标准偏差都和预期值相差甚远. 所幸的是, 在fastai中加入`Normalize`转换可以很容易实现归一化. 它在整个小批量一次性进行的, 所以可以在数据块建立时将它加到`batch_tfms`部分. 我们需要将要使用的均值和标准差传递给此变换; fastai附带了已经定义的标准ImageNet均值和标准差 (如果您没有将任何统计信息传递给Normalize转换, fastai将自动从单批数据中计算出它们).

我们添加这个转换(使用imagenet_stats, 因为Imagenette是ImageNet的子集), 现在看一批:

```
def get_dls(bs, size):
    block = DataBlock(blocks=(ImageBlock, CategoryBlock),
                              get_items=get_image_files,
                              get_y=parent_label,
                              item_tfms=Resize(460),
                              batch_tfms=[*aug_transforms(size=size, min_scale=0.75), Normalize.from_stats(*imagenet_stats)])
    return dblock.dataloaders(path, bs=bs)

```
```
dls = get_dls(64, 224)

x,y = dls.one_batch()
x.mean(dim=[0, 2, 3]), x.std(dim=[0, 2, 3])
```
>(TensorImage([-0.1637, -0.1425, -0.1583], device='cuda:0'),  
 TensorImage([1.2216, 1.2244, 1.2821], device='cuda:0'))

 看看这对模型训练起到了什么效果:

```
model = xresnet50(n_out=dls.c)
learn = Learn(dls, model, loss_func=CrossEntropyLossFlat(), metrics=accuracy)
learn.fit_one_cycle(5, 3e-3)
```
>
| epoch | train_loss | valid_loss | accuracy | time  |
| ----- | ---------- | ---------- | -------- | ----- |
| 0     | 1.596117   | 2.477990   | 0.442868 | 02:55 |
| 1     | 1.252596   | 1.167633   | 0.643764 | 02:53 |
| 2     | 0.960589   | 1.091942   | 0.663181 | 02:52 |
| 3     | 0.738749   | 0.695988   | 0.774832 | 02:54 |
| 4     | 0.594296   | 0.577121   | 0.816281 | 02:54 |

