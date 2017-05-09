import os
import datetime
import configparser
import tensorflow as tf
import layers as L
import numpy as np
import time
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

class Settings:
    def __init__(self, filename):
        print('Reading params form ' + filename)
        config = configparser.ConfigParser()
        config.read(filename)
        self.logdir = config.get('General', 'logdir')
        self.datapath = config.get('Train', 'data_file_path')


class Main:
    def __init__(self, sets):
        self.sets = sets
        self.size_image = 80
        self.n_channels = 3
        self.batch_size = 32
        self.learning_rate = 0.8
        self.epochs = 5000

        # net parameters
        self.logits = None
        self.cost = None
        # self.accuracy = None
        self.optimizer = None
        self.norm_probe = None
        self.filters = None
        self.accuracy = None

    def network_lenet(self, x):
        images = tf.reshape(x, shape=[-1, 28, 28, 1])
        # with tf.variable_scope('Conv_1') as scope:
        #     kernel = tf.get_variable('kernel', [5, 5, 1, 32],
        #                              initializer=tf.truncated_normal_initializer())
        #     biases = tf.get_variable('biases', [32],
        #                              initializer=tf.random_normal_initializer())
        #     conv = tf.nn.conv2d(images, kernel, strides=[1, 1, 1, 1], padding='SAME')
        #     conv1 = tf.nn.relu(conv + biases, name=scope.name)
        conv1 = L.conv(images, 'Conv_1', kw=5, kh=5, n_out=32)
        # with tf.variable_scope('Pool_1') as scope:
        #     pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        pool1 = L.pool(conv1, 'Pool_1', kh=2, kw=2)
        # with tf.variable_scope('Conv_2') as scope:
        #     kernel = tf.get_variable('kernel', [5, 5, 32, 64],
        #                              initializer=tf.truncated_normal_initializer())
        #     biases = tf.get_variable('biases', [64],
        #                              initializer=tf.random_normal_initializer())
        #     conv = tf.nn.conv2d(pool1, kernel, strides=[1, 1, 1, 1], padding='SAME')
        #     conv2 = tf.nn.relu(conv + biases, name=scope.name)
        conv2 = L.conv(pool1, 'Conv_2', kw=5, kh=5, n_out=64)
        # with tf.variable_scope('Pool_2') as scope:
        #     pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        pool2 = L.pool(conv2, 'Pool_2', kh=2, kw=2)

        # with tf.variable_scope('fc') as scope:
        #     shape = pool2.get_shape().as_list()
        #     dim = np.prod(shape[1:])
        #     tmp_tensor = tf.reshape(pool2, [-1, dim])
        #     # input_features = 7 * 7 * 64
        #     w = tf.get_variable('weights', [dim, 1024],
        #                         initializer=tf.truncated_normal_initializer())
        #     b = tf.get_variable('biases', [1024],
        #                         initializer=tf.random_normal_initializer())
        #     # pool2 = tf.reshape(pool2, [-1, input_features])
        #     fc = tf.nn.relu(tf.matmul(tmp_tensor, w) + b, name='relu')
        fc = L.flatten(pool2)
        fc = L.linear(fc, 1024, 'fc_1')
        fc = tf.nn.relu(fc, 'fc_1_relu')
        logits = L.linear(fc, 10, 'softmax_layer')

        #
        # with tf.variable_scope('softmax_linear') as scope:
        #     w = tf.get_variable('weights', [1024, 10],
        #                         initializer=tf.truncated_normal_initializer())
        #     b = tf.get_variable('biases', [10],
        #                         initializer=tf.random_normal_initializer())
        #     logits = tf.matmul(fc, w) + b
        return logits

    def main(self):
        launch_time = datetime.datetime.now()
        log = 'mni_{0:4}{1:2}{2:2}_{3:2}{4:2}{5:2}'.format(launch_time.year,
                                                           launch_time.month,
                                                           launch_time.day,
                                                           launch_time.hour,
                                                           launch_time.minute,
                                                           launch_time.second).replace(' ', '0')
        log_dir = os.path.join(self.sets.logdir, log)
        print('creating the net')
        with tf.name_scope('Inputs'):
            x = tf.placeholder(tf.float32, shape=[None, 784], name='x_input')
            y = tf.placeholder(tf.float32, shape=[None, 10], name='y_output')

        self.logits = self.network_lenet(x)
        # optimizer and loss
        with tf.name_scope('Crossentropy'):
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=y))

        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(y, 1))
        with tf.name_scope('Optimizer'):
            self.optimizer = tf.train.AdamOptimizer().minimize(loss=self.cost)
        with tf.name_scope('Accuracy'):
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar('loss', self.cost)
        tf.summary.scalar('accuracy', self.accuracy)
        merged_summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph(), flush_secs=10)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            print('Entering the session.')
            sess.run(init)
            # Start cycling epochs
            for i in range(10000):
                batch_x, batch_y = mnist.train.next_batch(self.batch_size)
                sess.run(self.optimizer, feed_dict={x: batch_x, y: batch_y})
                if i % 300 == 0:
                    acc, cost, summary = sess.run([self.accuracy, self.cost, merged_summary_op],
                                                  feed_dict={x: mnist.validation.images,
                                                             y: mnist.validation.labels})
                    summary_writer.add_summary(summary, i)
                    print('Accuracy for epoch (same validation)  {}: {}'.format(i, acc))






if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='list of possible arguments')
    parser.add_argument('-s', type=str, metavar='config', help="config file input", default='config.cfg')
    args = parser.parse_args()
    set = Settings(args.s)
    m = Main(set)
    m.main()
