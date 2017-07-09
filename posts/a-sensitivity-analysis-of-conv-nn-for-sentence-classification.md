title: CNNs 如何调参

---





> [A Sensitivity Analysis of Neural Networks for Sentence classification](https://arxiv.org/pdf/1510.03820v4.pdf)



### 概述

近来，CNNs对文本分类任务到达了不错的效果。No Free Lunch，要想获得一个好的效果，你得花时间做两件事：

1. model架构的设计
2. 参数的设定



CNNs调参很难，有2个主要原因：

1. 训练模型非常耗时
2. 架构设计和参数调整空间非常大，遍历所有组合是不可能的。



目前，也有很多调参技术，比如 random search， Baysian optimization。运用这些技术，首先也得需要知道参数取值的范围。



总之，什么架构，哪些参数对结果有影响，哪些是不重要的呢？我们最好能找出和数据无关，且对结果有很大影响的参数。该如何调这些参数，这篇论文做了一些分析，并总结一份guide。

<!--more-->

### baseline设定

1. 做参数分析、对比试验必备步骤，咱得找个参照物是不是。下图是各个数据集上，svm+不同特征对比。

![17-19-10](https://ws1.sinaimg.cn/large/006tNc79ly1fhdh1fsqf2j308w07sgmi.jpg)



2. 下图是CNN baseline的初始参数

![17-18-38](https://ws4.sinaimg.cn/large/006tNc79ly1fhdh1fbo9uj309e06mdgc.jpg)

调参步骤，只调一种参数，其他参数保持不变。



### 参数对比

#### 1. word2vec

word2vec出现，加深了深度学习方法在NLP中的应用，我们首先input word2vector 对结果的影响



![17-28-32](https://ws3.sinaimg.cn/large/006tNc79ly1fhdh1dj0suj310m0cy79q.jpg)

word2vec， Glove性能在不同数据集，表现不同。差别不大。 GloVe + Word2Vec组合对结果没有提升。



作者还实验了one-hot vector，结果显示对sentence级别的文本效果太差，原因是文本稀疏，数据量太少。



#### 2. filter region size



1.一种filter size

![17-50-01](https://ws3.sinaimg.cn/large/006tNc79ly1fhdh1hrvfkj30hy0c040c.jpg)

看下filter region size对结果的影响，下图结果是相比于baseline的。

![17-59-42](https://ws2.sinaimg.cn/large/006tNc79ly1fhdh1ecoluj30la0i2gn3.jpg)

**每份数据各有自己的最优region size， 这个参数需要我们去调的。一般sentence词数越多，region size越大。**



2. 多种filter size进行组合

![18-15-51](https://ws4.sinaimg.cn/large/006tNc79ly1fhdh1eu1iwj30m00futbo.jpg)

![18-16-14](https://ws4.sinaimg.cn/large/006tNc79ly1fhdh1cvbvbj30my0j6n0t.jpg)



**多种filter size的组合并不单种filter size要好。根据实验，我们首先找到单个最优的filter size， 然后在这个最优值附近选取其他值进行组合。**



#### filter number

![18-03-58](https://ws4.sinaimg.cn/large/006tNc79ly1fhdh1gu3sbj30me0gowfy.jpg)

可以看出 number of filter  对结果的影响取决于数据集。图表看出600以后，性能略有下降。我们知道，number of filter 越大，训练越耗时。一般范围在100到600。如果你的获得的最好结果在600附近，那么可以尝试下600以上的数据。



#### activation function

作者结论，Relu 和tanh较好。浅层的话， Iden（没有激活函数）也可以考虑。



#### pooling strategy



作者对比了max pooling 和 local pooling(equal sized local region on the feature map, **not entire feature map**)。最终 1-max pooling表现最好。

作者又对比了 k-max pooling， 结果显示1-max pooling最好。



#### regularization

作者实验表明 regularization对CNNs影响很少，作者解释 word2vec和bag of word 相比更能防止过拟合。 当number filter 比较大的时候，性能有所下降，这时可以考虑regularization。

### 总结



1. 考虑Table2的参数设定，如果数据集较大，可以考虑one-hot vector


2. 选出单个最后的filter size，一般1-10。 选出后，在最优值附近选几个值，进行组合调参
3. number future map 个数设在100-600
4. 尝试ReLU，tanh等activation function
5. 使用 1-max pooling
6. 当增加number feature map，性能有所下降时，dropout可以调高一点,>0.5





