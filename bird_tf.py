import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
# 读取数据


def get_files(filename):
    class_train = []
    label_train = []
    for train_class in os.listdir(filename):
        for pic in os.listdir(filename+train_class):
            name = train_class.split(sep='.')
            class_train.append(filename+train_class+'/'+pic)
            label_train.append(name[0])
    temp = np.array([class_train, label_train])
    temp = temp.transpose()
    # shuffle the samples
    np.random.shuffle(temp)
    # after transpose, images is in dimension 0 and label in dimension 1
    # 转换成list因为后面一些tensorflow函数（get_batch）接收的是list格式数据
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]
    # print(label_list)
    return image_list, label_list


# 产生用于训练的批次
# 生成相同大小的批次
def get_batch(image, label, image_W, image_H, batch_size, capacity):
    # image, label: 要生成batch的图像和标签list
    # image_W, image_H: 图片的宽高
    # batch_size: 每个batch有多少张图片
    # capacity: 队列容量
    # return: 图像和标签的batch

    # 将python.list类型转换成tf能够识别的格式
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # 生成队列
    input_queue = tf.train.slice_input_producer([image, label])

    image_contents = tf.read_file(input_queue[0])
    label = input_queue[1]
    image = tf.image.decode_jpeg(image_contents, channels=3)

    # 统一图片大小
    image = tf.image.resize_images(image, [image_H, image_W], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = tf.cast(image, tf.float32)
    # image = tf.image.per_image_standardization(image)   # 标准化数据
    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=64,  # 线程
                                              capacity=capacity)

    return image_batch, label_batch

# 定义训练的模型
# Alexnet
# def mmodel(x, keep_prob, num_classes):
#     # conv1
#     with tf.name_scope('conv1') as scope:
#         kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 96], dtype=tf.float32,
#                                                  stddev=1e-1), name='weights')
#         conv = tf.nn.conv2d(x, kernel, [1, 4, 4, 1], padding='SAME')
#         biases = tf.Variable(tf.constant(0.0, shape=[96], dtype=tf.float32),
#                              trainable=True, name='biases')
#         bias = tf.nn.bias_add(conv, biases)
#         conv1 = tf.nn.relu(bias, name=scope)
#
#     # pool1
#     pool1 = tf.nn.max_pool(conv1,
#                            ksize=[1, 3, 3, 1],
#                            strides=[1, 2, 2, 1],
#                            padding='VALID')
#
#     # conv2
#     with tf.name_scope('conv2') as scope:
#         kernel = tf.Variable(tf.truncated_normal([5, 5, 96, 256], dtype=tf.float32,
#                                                  stddev=1e-1), name='weights')
#         conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
#         biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
#                              trainable=True, name='biases')
#         bias = tf.nn.bias_add(conv, biases)
#         conv2 = tf.nn.relu(bias, name=scope)
#
#     # pool2
#     pool2 = tf.nn.max_pool(conv2,
#                            ksize=[1, 3, 3, 1],
#                            strides=[1, 2, 2, 1],
#                            padding='VALID',
#                            name='pool2')
#     # conv3
#     with tf.name_scope('conv3') as scope:
#         kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 384],
#                                                  dtype=tf.float32,
#                                                  stddev=1e-1), name='weights')
#         conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
#         biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
#                              trainable=True, name='biases')
#         bias = tf.nn.bias_add(conv, biases)
#         conv3 = tf.nn.relu(bias, name=scope)
#
#     # conv4
#     with tf.name_scope('conv4') as scope:
#         kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 384],
#                                                  dtype=tf.float32,
#                                                  stddev=1e-1), name='weights')
#         conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
#         biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
#                              trainable=True, name='biases')
#         bias = tf.nn.bias_add(conv, biases)
#         conv4 = tf.nn.relu(bias, name=scope)
#
#     # conv5
#     with tf.name_scope('conv5') as scope:
#         kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256],
#                                                  dtype=tf.float32,
#                                                  stddev=1e-1), name='weights')
#         conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
#         biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
#                              trainable=True, name='biases')
#         bias = tf.nn.bias_add(conv, biases)
#         conv5 = tf.nn.relu(bias, name=scope)
#
#     # pool5
#     pool5 = tf.nn.max_pool(conv5,
#                            ksize=[1, 3, 3, 1],
#                            strides=[1, 2, 2, 1],
#                            padding='VALID',
#                            name='pool5')
#
#     # flattened6
#     with tf.name_scope('flattened6') as scope:
#         flattened = tf.reshape(pool5, shape=[-1, 6 * 6 * 256])
#
#     # fc6
#     with tf.name_scope('fc6') as scope:
#         weights = tf.Variable(tf.truncated_normal([6 * 6 * 256, 4096],
#                                                   dtype=tf.float32,
#                                                   stddev=1e-1), name='weights')
#         biases = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32),
#                              trainable=True, name='biases')
#         fc6 = tf.nn.relu(tf.add(tf.matmul(flattened, weights), biases))
#
#     # dropout6
#     with tf.name_scope('dropout6') as scope:
#         dropout6 = tf.nn.dropout(fc6, keep_prob)
#
#     # fc7
#     with tf.name_scope('fc7') as scope:
#         weights = tf.Variable(tf.truncated_normal([4096, 4096],
#                                                   dtype=tf.float32,
#                                                   stddev=1e-1), name='weights')
#         biases = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32),
#                              trainable=True, name='biases')
#         fc7 = tf.nn.relu(tf.add(tf.matmul(dropout6, weights), biases))
#
#     # dropout7
#     with tf.name_scope('dropout7') as scope:
#         dropout7 = tf.nn.dropout(fc7, keep_prob)
#
#     # fc8
#     with tf.name_scope('fc8') as scope:
#         weights = tf.Variable(tf.truncated_normal([4096, num_classes],
#                                                   dtype=tf.float32,
#                                                   stddev=1e-1), name='weights')
#         biases = tf.Variable(tf.constant(0.0, shape=[num_classes], dtype=tf.float32),
#                              trainable=True, name='biases')
#         fc8= tf.nn.relu(tf.add(tf.matmul(dropout7, weights), biases))
#
#     return fc8
# smallmodel
def mmodel(images, batch_size):
    p = 0.5
    # conv1
    with tf.variable_scope('conv1') as scope: # 创建变量的作用域 若作用域为scope后面变量w的name为scope/w
        weights = tf.get_variable('weights',
                                  shape = [3,3,3, 16],
                                  dtype = tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(images, weights, strides=[1,1,1,1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        U1 = (np.random.rand(*conv.shape) < p) / p  # dropout
        pre_activation *= U1
        conv1 = tf.nn.relu(pre_activation, name= scope.name)
        # pool1
    with tf.variable_scope('pooling1_lrn') as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1],strides=[1,2,2,1],
                               padding='SAME', name='pooling1')
        # norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001/9.0,
        #                   beta=0.75,name='norm1')
        # conv2
    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3,3,16,128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(pool1, weights, strides=[1,1,1,1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        U1 = (np.random.rand(*conv.shape) < p) / p  # dropout
        pre_activation *= U1
        conv2 = tf.nn.relu(pre_activation, name='conv2')
        # pool2
    with tf.variable_scope('pooling2_lrn') as scope:
        # norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001/9.0,
        #                   beta=0.75,name='norm2')
        pool2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,1,1,1],
                               padding='SAME',name='pooling2')
        # fc1
    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(pool2, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable('weights',
                                  shape=[dim,4096],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[4096],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

        # regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
        # wloss_w1 = regularizer(local3)
        # tf.add_to_collection('losses', wloss_w1)

        # fc2
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('softmax_linear',
                                  shape=[4096, 200],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[200],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(local3, weights), biases, name='softmax_linear')
    return softmax_linear


# def loss(logits, label_batches):
#     cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label_batches)
#     cost = tf.reduce_mean(cross_entropy)
#     return cost
def loss(logits, labels):
    with tf.variable_scope("loss") as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                       labels=labels, name="xentropy_per_example")
        loss = tf.reduce_mean(cross_entropy, name="loss")
        tf.summary.scalar(scope.name + "loss", loss)
    return loss


# def get_accuracy(logits, labels):
#     acc = tf.nn.in_top_k(logits, labels, 1)  # top-5准确率
#     acc = tf.cast(acc, tf.float32)
#     acc = tf.reduce_mean(acc)
#     return acc
def get_accuracy(logits, labels):
    with tf.variable_scope("accuracy") as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name + "accuracy", accuracy)
    return accuracy


# def trainning(loss, learning_rate):
#     with tf.name_scope("optimizer"):
#         optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
#         global_step = tf.Variable(0, name="global_step", trainable=False)
#         train_op = optimizer.minimize(loss, global_step=global_step)
#     return train_op

def trainning(loss, lr):
    train_op = tf.train.GradientDescentOptimizer (lr).minimize(loss)  # 当数据量小时，加入var_list更新指定参数
    return train_op

data_dir = 'D:/v-zzx/train/'
test_dir = 'D:/v-zzx/test/'
log_dir = 'D:/v-zzx/log/'
# data_dir = 'D:/data/birddata/CUB_200_2011/train/'
# test_dir = 'D:/data/birddata/CUB_200_2011/test/'
# log_dir = 'D:/data/birddata/CUB_200_2011/log/'
size = 32
batch_size = 256
capacity = 256
num_classes = 200
learning_rate = 0.001
epoch = 10000
keep_prob = 0.5  # 随机失活概率

x = tf.placeholder(tf.float32, [batch_size, 32, 32, 3])
y_ = tf.placeholder(tf.int32, [batch_size])

image, label = get_files(data_dir)
image_batches, label_batches = get_batch(image, label, size, size, batch_size, capacity)
timage, tlabel = get_files(test_dir)
timage_batches, tlabel_batches = get_batch(timage, tlabel, size, size, batch_size, capacity)

q = mmodel(x,batch_size)
cost = loss(q, y_)
# 衰减的学习率
# global_step = tf.Variable(0, trainable=False)
# starter_learning_rate = 0.001
# learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
#                                            200, 0.96, staircase=True)
train_op = trainning(cost, learning_rate)
acc = get_accuracy(q, y_)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver()
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
Cross_loss = []
Test_acc = []
Train_acc = []


try:
    for step in np.arange(epoch):
        print(step)
        if coord.should_stop():
            break
        x_1, y_1 = sess.run([image_batches, label_batches])
        _, train_acc, train_loss = sess.run([train_op, acc, cost], feed_dict={x:x_1,y_:y_1})
        Cross_loss.append(train_loss)
        Train_acc.append(train_acc)
        print("loss:{} \ntrain_accuracy:{}".format(train_loss, train_acc))
        x_2, y_2 = sess.run([timage_batches, tlabel_batches])
        # print("test_accuracy:", sess.run(acc, feed_dict={x: x_2, y_:y_2}))
        test_acc = sess.run(acc, feed_dict={x: x_2, y_:y_2})
        Test_acc.append(test_acc)
        print("test_accuracy:{}".format(test_acc))
        if step % 1000 == 0:
            check = os.path.join(log_dir, "model.ckpt")
            saver.save(sess, check, global_step=step)
except tf.errors.OutOfRangeError:
    print("Done!!!")
finally:
    coord.request_stop()
coord.join(threads)
# 画出loss图像

fig, ax = plt.subplots(figsize=(13, 6))
ax.plot(Cross_loss)
plt.grid()
plt.title('Train loss')
plt.show()
fig, ax = plt.subplots(figsize=(13, 6))
ax.plot(Train_acc)
plt.grid()
plt.title('Train acc')
plt.show()
fig, ax = plt.subplots(figsize=(13, 6))
ax.plot(Test_acc)
plt.grid()
plt.title('Test acc')
plt.show()

sess.close()
