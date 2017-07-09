title: tensorflow cookbook -1

date: 2017/06/25 20:46:25

---



## tensorflow cookbook -1



## 工作流程

#### 1. 整理数据，或自动生成数据

寻找公共数据集，或者自动生成数据。然后normalize。最后把数据集分成 train，test， validation 等集合。

test验证模型在不同训练集上的效果，validation用于调参数。



####  2. 设定算法参数

比如 learning_rate, batch_size, epoch等



#### 3. 初始化 variables 和 placeholders

这两个区别是：`variables`一般是要训练的参数（weight & biase），tf自动调整其值。`placeholders`是来自样本或训练数据，在训练的时候，通过它喂给model训练。



#### 4. 定义model架构

例如Keras：

```python
from keras.models import Sequential
from keras.layers import Dense, Activation


model = Sequential()
model.add(Dense(units=64, input_dim=100))
model.add(Activation("relu"))
model.add(Dense(units=10))
model.add(Activation("softmax"))

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

```



在此推荐[neupy](https://github.com/itdxer/neupy),定义model架构更方便流畅。



#### 5.定义损失函数

损失函数评估model的输出，计算损失。训练时，优化器会根据这个损失进行参数的调整。

```python
loss = tf.reduce_mean(tf.square(y_actual - y_pred))
```



#### 6.开始训练model 

```python
with tf.Session(graph=graph) as sess:
    ...
    sess.run(...)
    ...
```



#### 7.评估model

用非训练数据，评测下model的效果



#### 8. 调参数

大部分时间，是来做这一步的，流程是重复上面几个步骤。





## Tensor

这个家伙很重要，相当于TensorFlow的血液。`varaible`和`placeholder`都用`Tensor`这个结构体存储。

my_var = tf.Variable(tensor)



##### Tensor创建

1. 指定维度创建

   ```python
   zero_tsr = tf.zeros([row_dim, col_dim])
   ones_tsr = tf.ones([row_dim, col_dim])
   filled_tsr = tf.fill([row_dim, col_dim], 42)
   constant_tsr = tf.constant([1,2,3])
   matrix_constant = tf.constant(42, [row_dim, col_dim])

   ```

2. copy已有tensor的维度

```python
zeros_similar = tf.zeros_like(constant_tsr)
ones_similar = tf.ones_like(constant_tsr)
```

3. 按区间定义tensor

```python
#not include the limit value, the result [6,9,12]
integer_seq_tsr = tf.range(start=6, limit=15, delta=3)
```

4. 随机tensor

```python
randunif_tsr = tf.random_uniform([row_dim, col_dim],
minval=0, maxval=1)
```



## 使用 variable，placeholders



```python
my_var = tf.Variable(tf.zeros([2,3]))
sess = tf.Session()
initialize_op = tf.global_variables_initializer ()
sess.run(initialize_op)

sess = tf.Session()
x = tf.placeholder(tf.float32, shape=[2,2])
y = tf.identity(x)
x_vals = np.random.rand(2,2)
sess.run(y, feed_dict={x: x_vals})

```



