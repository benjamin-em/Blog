## FastAI 第6章学习笔记

>和之前一样, 先安装和加载必要的库  

```
!pip install -Uqq fastbook
import fastbook
fastbook.setup_book()

from fastbook import *
```

### Other Computer Vision Problems

这一章看看多分标签分类问题和回归问题. 前者是预测一张图片中可能出现不止一个标签或者一个都没有, 后者不是类别做标签, 而是一个或几个数字.

### Multi-Label Classification

在之前的熊的分类器中, 这个可能很有用. 比如在第二章中,一个图片里一个熊都没有,但模型还是预测到了grizzly, black或teddy熊中的之中- 它没法预测“根本没有熊”的情况(在我自己的例子中,是区分动物,老虎狮子大象,没法预测图片中没有动物的情况). 多标签分类器(Multi-Label Classification)可以解决这个问题.

实际应用中,多标签分类这种应用项目不多, 但这种问题却经常碰到. 实际问题中,碰到0个或者多个预测出现在一张图片中的例子更普遍,所以我们因更多地考虑多标签分类问题.

先看看多标签数据集是什么样的.然后解释怎样让它为模型做好准备.模型的结构和上一章的没什么变化.只是损失函数变量.

#### The Data

我们用PASCAL数据集作为例子. 下载解压数据集:

```
from fastai.vision.all import *
path = untar_data(URLs.PASCAL_2007)
```

和之前以目录或文件名构成不一样的是,这个数据集是以CSV文件制定每个图像用什么标签.我们可以将CSV文件读到Pandas DataFrame来查看.

```
df = df.read_csv(path/'train.csv')
df.head()
```

| fname |     labels |     is_valid |       |
| ----: | ---------: | -----------: | ----- |
|     0 | 000005.jpg |        chair | True  |
|     1 | 000007.jpg |          car | True  |
|     2 | 000009.jpg | horse person | True  |
|     3 | 000012.jpg |          car | False |
|     4 | 000016.jpg |      bicycle | True  |

> 我们可以用```iloc```属性获取DataFrame的一列:
>
> ```
> df.iloc[:,0]
> ```
> >0       000005.jpg  
> >1       000007.jpg  
> >2       000009.jpg  
> >3       000012.jpg  
> >4       000016.jpg  
> >       ...  
> >5006    009954.jpg  
> >5007    009955.jpg  
> >5008    009958.jpg  
> >5009    009959.jpg  
> >5010    009961.jpg  
> >Name: fname, Length: 5011, dtype: obj
>
> ```
> df.iloc[0,:]
> # Trailing :s are always optional (in numpy, pytorch, pandas, etc.),
> #   so this is equivalent:
> df.iloc[0]
> ```
> >fname       000005.jpg
> >labels           chair
> >is_valid          True
> >Name: 0, dtype: object
>
> 新建列
> ```
> tmp_df = pd.DataFrame({'a':[1,2], 'b':[3,4]})
> tmp_df
> ```
>
> |      |    a |    b |
> | -- | -- | -- |
> |    0 |    1 |    3 |
> |    1 |    2 |    4 |
>
> ```
> tmp_df['c'] = tmp_df['a']+tmp_df['b']
> tmp_df
> ```
>
> |    a |    b |    c |      |
> | ---: | ---: | ---: | ---- |
> |    0 |    1 |    3 | 4    |
> |    1 |    2 |    4 | 6    |
> 
>Padas 是一个非常高效灵活的库,也是对数据科学家非常重要的一个工具.  [Python for Data Analysis](http://shop.oreilly.com/product/0636920023784.do) 这本书介绍了Padas，同时也包含```matplotlib```和```numpy```的介绍.

#### Constructing a DataBlock
怎样将一个DataFrame对象转换成DataLoaders 对象?  条件允许的话,一般建议用数据块API建立一个DataLoaders对象,因为它同时具备灵活性和易用性. 下面来介绍个这样的例子.
Pytoch和fastai有两个主要的类,用于表示和访问一个数据集或验证.

Dataset
     一个集合,返回的是单个数据项 - 从因变量和从变量构成的一个元组.
    
DataLoader
    一个提供小批量流的一个迭代器,每个小批量


[Back to contents page](index.md)

