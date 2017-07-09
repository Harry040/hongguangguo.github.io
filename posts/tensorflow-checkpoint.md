title: tensorflow checkpoint  用法

linenos: true

---



## 1. why tensorflow checkpoint？

场景1：

​	一个model从零开始训练，花了一天时间，loss稳定了。这时你想到另一个不错的loss function，实验时，难道还要从零开始吗？那太耗时了，这时就可以用你保存的checkpoint来初始化你的model。训练时间会大大减少



场景2：

​	你model训练到3000step时，你发现一个log日志忘了打印，或者一个小bug需要改等等。由于你保存了checkpoint，这时你可以毫不犹豫的kill掉进程，改代码修补。



场景3：

​	你的model训练了很多steps，比如10000 steps。但是哪一步是比较好的呢？这时你可以加载不同step的checkpoint进行评测。



场景4：

​	这个场景也是工作的需求，生产环境用的是c++，而model是用keras训练的。那怎么部署呢？很简单，直接`tf.train.Saver`保存，用c++可以直接加载。[参考链接](https://github.com/tensorflow/tensorflow/blob/ad50dafe8dfc9ee2b3e0f21d4a1ff37a80f220aa/tensorflow/python/training/saver.py#L876)

场景n:

​	你想到什么场景，可以邮件联系我，万分感谢！



<!--more-->

## 2. save tensorflow checkpoint

废话少说上code，tensorflow cookbook一段代码作为例子。

```python
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
# Classification Example
# We will create sample data as follows:
# x-data: sample 50 random values from a normal = N(-1, 1)
#         + sample 50 random values from a normal = N(1, 1)
# target: 50 values of 0 + 50 values of 1.
#         These are essentially 100 values of the corresponding output index
# We will fit the binary classification model:
# If sigmoid(x+A) < 0.5 -> 0 else 1
# Theoretically, A should be -(mean1 + mean2)/2

ops.reset_default_graph()

# Create graph
sess = tf.Session()

# Create data
x_vals = np.concatenate((np.random.normal(-1, 1, 50), np.random.normal(3, 1, 50)))
y_vals = np.concatenate((np.repeat(0., 50), np.repeat(1., 50)))
x_data = tf.placeholder(shape=[1], dtype=tf.float32)
y_target = tf.placeholder(shape=[1], dtype=tf.float32)

# Create variable (one model parameter = A)
A = tf.Variable(tf.random_normal(mean=10, shape=[1]))

# Add operation to graph
# Want to create the operstion sigmoid(x + A)
# Note, the sigmoid() part is in the loss function
my_output = tf.add(x_data, A)
#稍后说明
tf.add_collection('predict', my_output)

# Now we have to add another dimension to each (batch size of 1)
my_output_expanded = tf.expand_dims(my_output, 0)
y_target_expanded = tf.expand_dims(y_target, 0)

# Initialize variables
#init = tf.global_variables_initializer()
init = tf.initialize_all_variables()
sess.run(init)

# Add classification loss (cross entropy)
xentropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=my_output_expanded, labels=y_target_expanded)

# Create Optimizer
my_opt = tf.train.GradientDescentOptimizer(0.05)
train_step = my_opt.minimize(xentropy)
save = tf.train.Saver()
# Run loop
for i in range(1400):
    rand_index = np.random.choice(100)
    rand_x = [x_vals[rand_index]]
    rand_y = [y_vals[rand_index]]

    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    if (i+1)%200==0:
        print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)))
        print('Loss = ' + str(sess.run(xentropy, feed_dict={x_data: rand_x, y_target: rand_y})))
        #每200保存一次，默认保存6个checkpoints
        save.save(sess, './checkpoint_dir/checkpoint_model', global_step=i)
```

主要代码：

**save = tf.train.Saver()**

**save.save(sess, './checkpoint_dir/checkpoint_model', global_step=i)**



结果为：

```shell
ll ./checkpoint_dir
```

![13-07-09](https://ws4.sinaimg.cn/large/006tNc79ly1fhdjwqah5qj30ij08hjtx.jpg)



```shell
cat ./checkpoint_dir/checkpoint
```

![13-10-02](https://ws4.sinaimg.cn/large/006tNc79ly1fhdjzsdz0dj30b303gjrx.jpg)

上面是各个checkpoint的名字，很重要。后续的加载都是用的checkpoint名字。

## 3. load checkpoint

比如我想加载第999step的模型

```python
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
# Classification Example
# We will create sample data as follows:
# x-data: sample 50 random values from a normal = N(-1, 1)
#         + sample 50 random values from a normal = N(1, 1)
# target: 50 values of 0 + 50 values of 1.
#         These are essentially 100 values of the corresponding output index
# We will fit the binary classification model:
# If sigmoid(x+A) < 0.5 -> 0 else 1
# Theoretically, A should be -(mean1 + mean2)/2

ops.reset_default_graph()

# Create graph
sess = tf.Session()

# Create data
x_vals = np.concatenate((np.random.normal(-1, 1, 50), np.random.normal(3, 1, 50)))
y_vals = np.concatenate((np.repeat(0., 50), np.repeat(1., 50)))
x_data = tf.placeholder(shape=[1], dtype=tf.float32)
y_target = tf.placeholder(shape=[1], dtype=tf.float32)

# Create variable (one model parameter = A)
A = tf.Variable(tf.random_normal(mean=10, shape=[1]))

# Add operation to graph
# Want to create the operstion sigmoid(x + A)
# Note, the sigmoid() part is in the loss function
my_output = tf.add(x_data, A)
#稍后说明
tf.add_collection('predict', my_output)

# Now we have to add another dimension to each (batch size of 1)
my_output_expanded = tf.expand_dims(my_output, 0)
y_target_expanded = tf.expand_dims(y_target, 0)

# Initialize variables
#init = tf.global_variables_initializer()
init = tf.initialize_all_variables()
sess.run(init)

# Add classification loss (cross entropy)
xentropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=my_output_expanded, labels=y_target_expanded)

# Create Optimizer
my_opt = tf.train.GradientDescentOptimizer(0.05)
train_step = my_opt.minimize(xentropy)
save = tf.train.Saver()

#从999的checkpoint恢复
save.restore(sess, './checkpoint_dir/checkpoint_model-999')
# Run loop
for i in range(1400):
    rand_index = np.random.choice(100)
    rand_x = [x_vals[rand_index]]
    rand_y = [y_vals[rand_index]]

    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    if (i+1)%200==0:
        print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)))
        print('Loss = ' + str(sess.run(xentropy, feed_dict={x_data: rand_x, y_target: rand_y})))
        #每200保存一次，默认保存6个checkpoints
        save.save(sess, './checkpoint_dir/checkpoint_model', global_step=i)
```



重要代码：

`save.restore(sess, './checkpoint_dir/checkpoint_model-999')`





## 4. load pretrained checkpoint

这个大部分用在production中，比如你用tf python，或keras写的模型，怎么部署到线上呢？很简单,训练完毕后用`tf.train.Saver`保存模型，在生产环境支持的代码加载这个model就行了（包括图结构，和对应的权重）。

```py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
# Classification Example
# We will create sample data as follows:
# x-data: sample 50 random values from a normal = N(-1, 1)
#         + sample 50 random values from a normal = N(1, 1)
# target: 50 values of 0 + 50 values of 1.
#         These are essentially 100 values of the corresponding output index
# We will fit the binary classification model:
# If sigmoid(x+A) < 0.5 -> 0 else 1
# Theoretically, A should be -(mean1 + mean2)/2

def train():
    # Create graph
    sess = tf.Session()

    # Create data
    x_vals = np.concatenate((np.random.normal(-1, 1, 50), np.random.normal(3, 1, 50)))
    y_vals = np.concatenate((np.repeat(0., 50), np.repeat(1., 50)))
    x_data = tf.placeholder(shape=[1], dtype=tf.float32, name='input')
    y_target = tf.placeholder(shape=[1], dtype=tf.float32)

    # Create variable (one model parameter = A)
    A = tf.Variable(tf.random_normal(mean=10, shape=[1]))

    # Add operation to graph
    # Want to create the operstion sigmoid(x + A)
    # Note, the sigmoid() part is in the loss function
    my_output = tf.add(x_data, A)
    predict = tf.round(tf.sigmoid(my_output))
    tf.add_to_collection('predict_op', predict)
    # Now we have to add another dimension to each (batch size of 1)
    my_output_expanded = tf.expand_dims(my_output, 0)
    y_target_expanded = tf.expand_dims(y_target, 0)

    # Initialize variables
    #init = tf.global_variables_initializer()
    init = tf.initialize_all_variables()
    sess.run(init)

    # Add classification loss (cross entropy)
    xentropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=my_output_expanded, labels=y_target_expanded)

    # Create Optimizer
    my_opt = tf.train.GradientDescentOptimizer(0.05)
    train_step = my_opt.minimize(xentropy)
    save = tf.train.Saver()
    # Run loop
    for i in range(1400):
        rand_index = np.random.choice(100)
        rand_x = [x_vals[rand_index]]
        rand_y = [y_vals[rand_index]]

        sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
        if (i+1)%200==0:
            print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)))
            print('Loss = ' + str(sess.run(xentropy, feed_dict={x_data: rand_x, y_target: rand_y})))
            save.save(sess, './checkpoint_dir/checkpoint_model', global_step=i)

def predict():
    sess = tf.Session()
    saver = tf.train.import_meta_graph('./checkpoint_dir/checkpoint_model-999.meta')
    saver.restore(sess, './checkpoint_dir/checkpoint_model-999')
    graph = tf.get_default_graph()
    predict_op = tf.get_collection('predict_op')[0]
    x_data = graph.get_tensor_by_name('input:0')
    print sess.run(predict_op, {x_data: [2.4]})

if __name__ == '__main__':
    train()
    predict()
```



重点是63-69行， 比如你同事训练好一个模型`checkpoint_model-999`。你用时，直接加载事先定义的op即可。

比如67，68行。 `tf.get_collection`操作需要事先在train中用`tf.add_to_collection`定义。









---------

=======================害羞的分割线================

如对您有帮助，请赐我钱进的动力吧！《：

![ok](/Users/thinkdeeper/Downloads/ok.jpg)