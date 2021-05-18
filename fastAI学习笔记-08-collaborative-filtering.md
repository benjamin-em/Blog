## FastAI 第8章 - Collaborative Filtering Deep Dive

>先安装和加载必要的库  

```
!pip install -Uqq fastbook
import fastbook
fastbook.setup_book()

from fastbook import *
```

这章讲的是协同过滤, 推荐系统.  例如推荐奈飞上的电影, 要举出一个用户的主页上突出显示的电影是什么; 在社交媒体上提供显示哪些故事等待. 这类问题有一个通用的解决方案, 叫做协同过滤 - *collaborative filtering* , 它的工作原理是, 观察目前用户用过或者喜欢的产品, 查找其他用户用过或喜欢的相似的产品, 然后推荐被人用过或喜欢的其他产品. 

举个例子, 在奈飞上你可能看过很多科幻动作片, 并且都是1970年代的拍的. 奈飞也许不知道这些你看过的电影的特定属性, 但是它可以知道其他看过相同电影的人还倾向于哪些其他1970年代的科幻动作电影.  换句话说, 用这种方法, 我们不必知道关于电影的任何信息, 除了谁会喜欢看. 






[Back to contents page](index.md)

