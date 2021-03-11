## FastAI 第5章学习笔记

### Image Classification
现在深入一点，什么是计算机视觉模型(computer vision model), NLP模型, tabular模型等等？怎样构建一个符合你特定领域需求的框架？
怎样从训练过程中得到尽可能最好的结果？怎样处理更快?当数据集发生变化时，我们应该做怎样的修改？现在在第一章例子的基础上做两件事：  
- 优化
- 使其在更多种类的数据上得到应用

为此,我们需要学会：各种layers(层),正则化方法,优化器,怎样把各层放到一个架构中,标记方法技巧等等.

### From Dogs and Cats to Pet Breeds

第一个模型演示了怎样区分狗子和猫子. 就在几年前,这是一个很大的挑战,但现在so easy！而且事实证明,同样的数据集允许我们解决一些更有
挑战性的问题：指出途中宠物的品种.  
网站上已经下载了“宠物”的数据集.
```
from fastai.vision.all import *
path = untar_data(URLs.PETS)
```
通过python的```untar```函数解压到本地,可以看看```URLs.PETS```的链接和```path```这个本地路径
```
URLs.PETS
```
> 'https://s3.amazonaws.com/fast-ai-imageclas/oxford-iiit-pet.tgz'

```
path
```
> Path('/root/.fastai/data/oxford-iiit-pet')

数据布局是深度学习难题的重要组成部分,数据一般以这两种方式提供：  
- 每个文件表示一项数据,例如文本文件,图像,可能是以目录归类或者文件名的形式表示这些项目的信息.
- 一个数据表,例如以CSV的形式,里面每行是一项数据,可能包含文件名,与其他形式如图片文档相关联.

在特定领域,也有一些例外,如在基因学中,可以是二进制的数据库甚至可以是网络流.但绝大多数领域还是上面那两种,或两种的结合.
用```ls```方法看下数据集里面有什么
```
Path.BASE_PATH = path
path.ls()
```
> (#2) [Path('annotations'),Path('images')]

可以看到有annotations 和 images 两个目录. 数据集的[官网](https://www.robots.ox.ac.uk/~vgg/data/pets/)说annotations目录包含
宠物的分部地区而不是宠物是什么.我们现在要辨认的是宠物品种,而不是地区分部,所以不必关心这个目录.所以看看images里有什么：
```
(path/"images").ls()
```
>(#7393) [Path('images/japanese_chin_131.jpg'),Path('images/Bombay_4.jpg'),Path('images/Birman_43.jpg'),Path('images/Maine_Coon_57.jpg'),Path('images/pug_80.jpg'),Path('images/english_cocker_spaniel_140.jpg'),Path('images/american_bulldog_175.jpg'),Path('images/boxer_154.jpg'),Path('images/saint_bernard_104.jpg'),Path('images/wheaten_terrier_59.jpg')...]

FastAI中很多函数和方法返回集合时会使用```L```类,```L```可以看做一个加强版的python```list```类型,相比```list```，它会提供一些加强
功能,例如在列表的前面会加```#```号加数字来表示列表中元素的个数,当列表太长时,会在后面用省略号.
