## FastAI 第2章学习笔记  
刚开始学习深度学习，从[fastAI的入门课程](https://course.fast.ai/)开始,这个课程很适合小白（虽然要学完对我来说也是个巨大的工程…），
也有[论坛](https://forums.fast.ai/c/part1-v4/46)
目前学到视频第三节，回过头来做[第二章](https://colab.research.google.com/github/fastai/fastbook/blob/master/02_production.ipynb)的笔记,算是复习了。

### 使用duckduckGo搜索图片

书中介绍了使用bing搜索图片，作为数据集来源。但是必应注册API过程太麻烦，尤其是需要信用卡,于是使用书中介绍的[第二种方法](https://course.fast.ai/images),
duckduckGo.

不过fastAI 中search_images_ddg() 这个API似乎有bug，没办法只能再次迂回。终于找到论坛中找到[一种解决方案](https://forums.fast.ai/t/creating-image-datasets-for-vision-learning/77673/2)  
先安装依赖包
```
### 文中代码时以Jupyter 写的，以!开头代表直接在jupyter页面执行shell命令
Path().cwd()
!rm gdrive/MyDrive/images -rf
!pip install -q jmd_imagescraper
```

使用jmd_imagescraper下载图片
```
from jmd_imagescraper.core import *
root = Path().cwd()/"gdrive/MyDrive/images"

animal_types = ('tiger', 'lion', 'elephant', 'giraffe', 'panda')

for o in animal_types:
  duckduckgo_search(root, o, o, max_results=300)
```
