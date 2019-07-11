import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt
import tensorflow.contrib.slim as slim

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# 读取数据
mnist = input_data.read_data_sets('./MNIST_data/', one_hot=True)

# 查看数据
print("训练数据：", mnist.train.images.shape)
print("训练数据标签：", mnist.train.labels.shape)
print("验证数据：", mnist.validation.images.shape)
print("验证数据标签：", mnist.validation.labels.shape)
print("测试数据：", mnist.test.images.shape)
print("测试数据标签：", mnist.test.labels.shape)
print("测试数据标签：", mnist.test.labels.shape)

# 查看训练集中的钱十六张图片
plt.figure(figsize=(8, 8))
for idx in range(16):
    plt.subplot(4, 4, idx + 1)
    plt.axis('off')
    plt.title('[{}]'.format(np.argmax(mnist.train.labels[idx])))
    plt.imshow(mnist.train.images[idx].reshape((28, 28)))

# 定义训练网络的输入
x = tf.placeholder("float", [None, 784], name='x')
y = tf.placeholder("float", [None, 10], name='y')
# 将一维数组还原成二维图片
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 定义第一个卷积层 卷积核：6个5*5；padding：valid
# 第一层卷积层的输出：宽高为24*24；深度为6
with tf.name_scope('conv1'):
    C1 = slim.conv2d(x_image, 6, [5, 5], padding='SAME', activation_fn=tf.nn.relu)

# 定义第一层池化 stride=2
# 池化后输出：深度不变，但是宽高减半。即宽高为12*12；深度为6
with tf.name_scope('pool1'):
    S2 = slim.max_pool2d(C1, [2, 2], stride=[2, 2], padding='SAME')

# 定义第二层卷积层 卷积核：16个5*5；padding：valid
# 第二层卷积层的输出：宽高为8*8；深度为16
with tf.name_scope('conv2'):
    C3 = slim.conv2d(S2, 32, [5, 5], padding='SAME', activation_fn=tf.nn.relu)

# 定义第二层池化 stride=2
# 池化后的输出：宽高4*4；深度为16
with tf.name_scope('pool2'):
    S4 = slim.max_pool2d(C3, [2, 2], stride=[2, 2], padding='SAME')

# 池化后的数据是三维的，将三维数据变成一维数据，然后送入两层全连接网络，全连接隐层中的神经元个数分别为120， 84
with tf.name_scope('fc1'):
    S4_flat = slim.flatten(S4)
    C5 = slim.fully_connected(S4_flat, 120, activation_fn=tf.nn.relu, weights_regularizer=tf.contrib.layers.l2_regularizer(0.0001))
with tf.name_scope('fc2'):
    F6 = slim.fully_connected(C5, 84, activation_fn=tf.nn.relu, weights_regularizer=tf.contrib.layers.l2_regularizer(0.0001))

# 对特征添加一个0.6的Dropout，以减少过拟合
# dropout仅在训练时使用，验证的时候需要关闭dropout，所以验证时候的keep_prob=1.0
# dropout的输出最终送入一个隐层为10的全连接层，这个全连接层即为最后的分类器
with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(name='keep_prob', dtype=tf.float32)
    F6_drop = tf.nn.dropout(F6, keep_prob)
with tf.name_scope('fc3'):
    logits = slim.fully_connected(F6_drop, 10, activation_fn=None, weights_regularizer=tf.contrib.layers.l2_regularizer(0.0001))

# 定义loss和优化器 loss计算使用：sparse_softmax_cross_entropy_with_logits 优化器：sgd 学习率：0.3
cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])
for w in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
    print(w.name)
    tf.summary.histogram(w.name, w)

total_loss = cross_entropy_loss + 7e-5 * l2_loss
tf.summary.scalar('cross_entropy_loss', cross_entropy_loss)
tf.summary.scalar('l2_loss', l2_loss)
tf.summary.scalar('total_loss', total_loss)

lr = tf.train.exponential_decay(0.03, 5500, 100, 0.96, staircase=True)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(total_loss)

# 对网络进行softmax激活，得到概率分布
pred = tf.nn.softmax(logits)
correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(pred, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# saver用于保存或恢复训练模型
batch_size = 100
training_step = 5500
saver = tf.train.Saver()

# 以上仅仅是定义了网络结构，下面创建Session，并将数据填入网络运行
merged = tf.summary.merge_all()
with tf.Session() as sess:
    writer = tf.summary.FileWriter("./logs/", sess.graph)
    sess.run(tf.global_variables_initializer())

    # 定义验证数据集和测试数据集
    validate_data = {
        x: mnist.validation.images,
        y: mnist.validation.labels,
        keep_prob: 1.0
    }
    test_data = {
        x: mnist.test.images,
        y: mnist.test.labels,
        keep_prob: 1.0
    }

    for i in range(training_step):
        xs, ys = mnist.train.next_batch(batch_size)
        _, loss, rs = sess.run([optimizer, cross_entropy_loss, merged],
                               feed_dict={x: xs, y: ys, keep_prob: 0.5})
        writer.add_summary(rs, i)

        # 每100次训练打印一次损失值与验证准确率
        if i > 0 and i % 100 == 0:
            validate_accuracy = sess.run(accuracy, feed_dict=validate_data)
            print("after %d training steps, the loss is %g, the validation accuracy is %g" %
                  (i, loss, validate_accuracy))
            saver.save(sess, './model.ckpt', global_step=i)

    print("The training is finish!")
    print(lr.eval())
    # 最终的测试准确率
    acc = sess.run(accuracy, feed_dict=test_data)
    print("The test accuracy is : ", acc)

plt.show()
