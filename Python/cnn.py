from skimage import io, transform
import glob
import os
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
import random

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

'''
数据集来源
@misc{e-VDS,
      author = {Culurciello, Eugenio and Canziani, Alfredo},
               title = {{e-Lab} Video Data Set},
howpublished = {url{https://engineering.purdue.edu/elab/eVDS/}},
year={2017}
}
'''

# 训练数据
train_path = 'F:/10-image-set/train/'
# 验证数据
val_path = 'F:/10-image-set/val/'

# 将所有的图片resize成100*100
w = 100
h = 100
# 三色通道
c = 3


# 读取图片
def read_img(path):
    # 将path路径下的所有文件夹路径存到cate
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    imgs = []
    labels = []
    for idx, folder in enumerate(cate):
        print('分类标签idx:', idx)
        print('文件夹路径folder:', folder)
        i = 0
        # 将文件夹下所有图片挨个打上标签并分别存到imgs,labels
        files = glob.glob(folder + '/*.jpg')
        random.shuffle(files)
        for im in files:
            img = io.imread(im)
            img = transform.resize(img, (w, h, c), mode="reflect")
            imgs.append(img)
            # 以目录的index作为label
            labels.append(idx)
            # 数据集太大每种类别选6000张
            i += 1
            if i == 6500:
                break
    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)


# 加载训练数据集
print("加载训练数据集")
t_data, t_label = read_img(train_path)
# 加载验证数据集
print("加载验证数据集")
v_data, v_label = read_img(val_path)

# 打乱训练数据集的顺序
num_example = t_data.shape[0]
arr = np.arange(num_example)
np.random.shuffle(arr)
train_data = t_data[arr]
train_label = t_label[arr]

# 训练集
s = np.int(num_example)
x_train = train_data[:s]
y_train = train_label[:s]
# 验证集
x_val = v_data
y_val = v_label

# -----------------构建网络----------------------
# shape=[None,  w, h, c]表示每个样本w*h*c维的向量表示，但不确定有多少个训练样本。所以第一维是None
# 样本的特征输入
x = tf.placeholder(tf.float32, shape=[None, w, h, c], name='x')
# 样本的类别标签
y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')

# 第一个卷积层（100->50)
# Tensorflow中padding有两种类型SAME和VALID SAME填充0使维度保持不变 VALID不填充0
conv1 = tf.layers.conv2d(
    inputs=x,
    filters=32,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu,
    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

# 第二个卷积层(50->25)
conv2 = tf.layers.conv2d(
    inputs=pool1,
    filters=64,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu,
    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

# 第三个卷积层(25->12)
conv3 = tf.layers.conv2d(
    inputs=pool2,
    filters=128,
    kernel_size=[3, 3],
    padding="same",
    activation=tf.nn.relu,
    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

# 第四个卷积层(12->6)
conv4 = tf.layers.conv2d(
    inputs=pool3,
    filters=128,
    kernel_size=[3, 3],
    padding="same",
    activation=tf.nn.relu,
    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

re1 = tf.reshape(pool4, [-1, 6 * 6 * 128])

# dropout减少过拟合
keep_prob1 = tf.placeholder(tf.float32, name="keep_prob1")
# tf.nn.dropout(x, keep_prob, noise_shape=None, seed=None, name=None)接口
# x:输入tensor  keep_prob:float类型，每个元素被保留下来的概率  noise_shape:一个1维的int32张量，代表了随机产生“保留/丢弃”标志的shape
# seed:整型变量，随机数种子  name:可视化时显示的名称
fc_drop1 = tf.nn.dropout(re1, keep_prob1)

# 全连接层
dense1 = tf.layers.dense(inputs=fc_drop1,
                         units=1024,
                         activation=tf.nn.relu,
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                         kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))

# dropout减少过拟合
keep_prob2 = tf.placeholder(tf.float32, name="keep_prob2")
# tf.nn.dropout(x, keep_prob, noise_shape=None, seed=None, name=None)接口
# x:输入tensor  keep_prob:float类型，每个元素被保留下来的概率  noise_shape:一个1维的int32张量，代表了随机产生“保留/丢弃”标志的shape
# seed:整型变量，随机数种子  name:可视化时显示的名称
fc_drop2 = tf.nn.dropout(dense1, keep_prob2)

dense2 = tf.layers.dense(inputs=fc_drop2,
                         units=512,
                         activation=tf.nn.relu,
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                         kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))

# dropout减少过拟合
keep_prob3 = tf.placeholder(tf.float32, name="keep_prob3")
# tf.nn.dropout(x, keep_prob, noise_shape=None, seed=None, name=None)接口
# x:输入tensor  keep_prob:float类型，每个元素被保留下来的概率  noise_shape:一个1维的int32张量，代表了随机产生“保留/丢弃”标志的shape
# seed:整型变量，随机数种子  name:可视化时显示的名称
fc_drop3 = tf.nn.dropout(dense2, keep_prob3)

# 最后输出层使用10个神经元得到10维向量对应分类
logits = tf.layers.dense(inputs=fc_drop3,
                         units=10,
                         activation=None,
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                         kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
# ---------------------------网络结束---------------------------

# 计算神经网络损失
# logits是神经网络最后一层的输入 labels是神经网络期望的输出
# 函数的作用就是计算最后一层的cross entropy，只不过tensorflow把softmax计算与cross entropy计算放到一起用一个函数来实现，用来提高程序的运行速度
loss = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=logits)

# 基于一定的学习率进行梯度优化训练
# tf.train.AdamOptimizer使用Adam算法的Optimizer
# 使用minimize()操作，该操作不仅可以计算出梯度，而且还可以将梯度作用在变量上
train_op = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(loss)

# 评估分类结果
# tf.argmax(input, dimension, name=None) - input：输入的张量 - demension - name：给部件起个名字，可以不用起
# 这个函数的作用是给出预测出来的5个结果（对应的每个类别的分类概率）中的最大值的下标
# correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), y_) 会生成一组向量，如：[True, False, True, True]
correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), y_)
# 把它映射成浮点数，然后计算它们的均值
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# 定义一个函数，按批次取数据
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]


# 训练和测试数据，可将n_epoch设置更大一些
# 训练次数
n_epoch = 15
# 每次取多少数据进行训练或验证
batch_size = 16

# 生成交互式会话
sess = tf.InteractiveSession()

# 加载模型进当前会话
saver = tf.train.Saver()
saver.restore(sess, 'E:/RealTimeIR/model/10-image-set')

# 从已保存的模型中就已经加载好这些参数继续训练
# 一次把所有的Variable类型初始化
# sess.run(tf.global_variables_initializer())

# 保存模型
# saver = tf.train.Saver(max_to_keep=1)

max_acc = 0
# 记录作图数据
plt_x = range(0, n_epoch)
plt_a1 = []
plt_a2 = []
plt_l1 = []
plt_l2 = []
for epoch in range(n_epoch):
    # training
    t_start_time = time.time()
    train_loss, train_acc, n_batch = 0, 0, 0
    for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size, shuffle=True):
        _, err, ac = sess.run([train_op, loss, acc],
                              feed_dict={x: x_train_a, y_: y_train_a, keep_prob1: 0.3, keep_prob2: 0.3,
                                         keep_prob3: 0.3})
        train_loss += err
        train_acc += ac
        n_batch += 1
    t_end_time = time.time()
    print("train loss: %f" % (train_loss / n_batch))
    plt_l1.append(train_loss / n_batch)
    print("train acc: %f" % (train_acc / n_batch))
    plt_a1.append(train_acc / n_batch)
    print("train time: %f 分钟" % divmod(t_end_time - t_start_time, 60)[0])

    # validation
    # 验证时间很短不记录时间
    val_loss, val_acc, n_batch = 0, 0, 0
    for x_val_a, y_val_a in minibatches(x_val, y_val, batch_size, shuffle=False):
        err, ac = sess.run([loss, acc],
                           feed_dict={x: x_val_a, y_: y_val_a, keep_prob1: 1.0, keep_prob2: 1.0, keep_prob3: 1.0})
        val_loss += err
        val_acc += ac
        n_batch += 1
    print("validation loss: %f" % (val_loss / n_batch))
    plt_l2.append(val_loss / n_batch)
    print("validation acc: %f" % (val_acc / n_batch))
    plt_a2.append(val_acc / n_batch)

    # 保存精确度最大的一次模型
    if val_acc > max_acc:
        max_acc = val_acc
        saver.save(sess, './model/10-image-set')
        print("模型保存，精度: %f" % (val_acc / n_batch))

# matplotlib作图
plt.figure("Accuracy&Loss")
# 第一张图展示Accuracy
plt.subplot(1, 2, 1)
plt.plot(plt_x, plt_a1, label='train accuracy', marker='o',
         markerfacecolor='blue', markersize=12)
plt.plot(plt_x, plt_a2, label='validation accuracy', linewidth=3, color='r', marker='o',
         markerfacecolor='red', markersize=12)
plt.xlabel('迭代次数')
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.legend()
# 第二张图展示Loss
plt.subplot(1, 2, 2)
plt.plot(plt_x, plt_l1, label='train loss', marker='o',
         markerfacecolor='blue', markersize=12)
plt.plot(plt_x, plt_l2, label='validation loss', linewidth=3, color='r', marker='o',
         markerfacecolor='red', markersize=12)
plt.xlabel('迭代次数')
plt.ylabel('Loss')
plt.title('Loss')
plt.legend()
plt.savefig("Accuracy&Loss.png")
# plt.show()

sess.close()
