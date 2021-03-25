## fastAI第一章学习笔记

这是第二篇学习笔记博客，但记录的是[fastAI课程](https://course.fast.ai/)第一章的学习笔记.
这里记录一些概念/术语的理解。

### 术语

| Term | Meaning
| ---- | ---- 
|Label | The data that we're trying to predict, such as "dog" or "cat"
|Architecture | The _template_ of the model that we're trying to fit; the actual mathematical function that we're passing the input data and parameters to
|Model | The combination of the architecture with a particular set of parameters
|Parameters | The values in the model that change what task it can do, and are updated through model training
|Fit | Update the parameters of the model such that the predictions of the model using the input data match the target labels
|Train | A synonym for _fit_
|Pretrained model | A model that has already been trained, generally using a large dataset, and will be fine-tuned
|Fine-tune | Update a pretrained model for a different task
|Epoch | One complete pass through the input data
|Loss | A measure of how good the model is, chosen to drive training via SGD
|Metric | A measurement of how good the model is, using the validation set, chosen for human consumption
|Validation set | A set of data held out from training, used only for measuring how good the model is
|Training set | The data used for fitting the model; does not include any data from the validation set
|Overfitting | Training a model in such a way that it _remembers_ specific features of the input data, rather than generalizing well to data not seen during training
|CNN | Convolutional neural network; a type of neural network that works particularly well for computer vision tasks

## 疑问
**关于fine tune这里有个问题，根据定义，fine tune是对一个已经训练好的模型进行“微调”，用于一个 _不同_ 的任务, 但是很多地方在训练之后，使用模型之前也会进行fine tune**
 如[第二章](https://colab.research.google.com/github/fastai/fastbook/blob/master/02_production.ipynb)中(当然第一章也有很多这样的例子)
 
```
bears = DataBlock(
    blocks=(ImageBlock, CategoryBlock), # what kind of date we want to working with - image
    get_items=get_image_files, #how to get the items - by files
    splitter=RandomSplitter(valid_pct=0.2, seed=42), #how to create validation set
    get_y=parent_label, # how to label these items
    item_tfms=Resize(128))
 
bears = bears.new(
    item_tfms=RandomResizedCrop(224, min_scale=0.5),
    batch_tfms=aug_transforms())
dls = bears.dataloaders(path)

learn = cnn_learner(dls, resnet18, metrics=error_rate) 

```
以上cnn_learner就是对模型进行针对这个任务(识别熊的种类)进行训练，后面也是用到该任务，下面的fine tune 是否有必要？
```
learn.fine_tune(4)
```

[Back to contents page](index.md)
