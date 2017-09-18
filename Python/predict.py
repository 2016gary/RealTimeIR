from skimage import io, transform
import glob
import tensorflow as tf
import numpy as np

path = 'E:/RealTimeIR/predict/'

# 将所有的图片resize成128*128
w = 100
h = 100
c = 3


# 读取图片
def read_img(path):
    imgs = []
    for im in glob.glob(path + '*.jpg'):
        img = io.imread(im)
        img = transform.resize(img, (w, h, c), mode="reflect")
        imgs.append(img)
    return np.asarray(imgs, np.float32)


# 将预测图片转为数据集
x_train = read_img(path)

# -----------------使用跟模型一致的网络----------------------
x = tf.placeholder(tf.float32, shape=[None, w, h, c], name='x')

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

# 全连接层
dense1 = tf.layers.dense(inputs=re1,
                         units=1024,
                         activation=tf.nn.relu,
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                         kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
dense2 = tf.layers.dense(inputs=dense1,
                         units=512,
                         activation=tf.nn.relu,
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                         kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))

# 最后输出层使用10个神经元得到10维向量对应分类
logits = tf.layers.dense(inputs=dense2,
                         units=10,
                         activation=None,
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                         kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
# ---------------------------网络结束---------------------------

sess = tf.InteractiveSession()
# 加载模型进当前会话
saver = tf.train.Saver()
saver.restore(sess, 'E:/RealTimeIR/model/10-image-set')
# 使用模型进行预测
predictions = sess.run(tf.argmax(logits, 1), feed_dict={x: x_train})
# print("输出predictions：", predictions)
for predict in predictions:
    if predict == 0:
        result = "单车"
        print("识别结果：单车")
    elif predict == 1:
        result = "书"
        print("识别结果：书")
    elif predict == 2:
        result = "水瓶"
        print("识别结果：水瓶")
    elif predict == 3:
        result = "汽车"
        print("识别结果：汽车")
    elif predict == 4:
        result = "椅子"
        print("识别结果：椅子")
    elif predict == 5:
        result = "电脑"
        print("识别结果：电脑")
    elif predict == 6:
        result = "人脸"
        print("识别结果：人脸")
    elif predict == 7:
        result = "鞋子"
        print("识别结果：鞋子")
    elif predict == 8:
        result = "桌子"
        print("识别结果：桌子")
    elif predict == 9:
        result = "树"
        print("识别结果：树")
    else:
        result = "识别错误"
        print("识别错误")

file_object = open('E:/RealTimeIR/result.txt', 'w+')
# 清空文件内容
file_object.truncate()
file_object.write(result)
file_object.close()

sess.close()
