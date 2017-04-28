import os
import numpy as np
import tensorflow as tf
import configparser
import glob
from scipy import misc
import matplotlib.pyplot as plt
import h5py
import time
import layers as L
import tools as T
import random as rand
from tqdm import tqdm


class Settings:
    def __init__(self, filename):
        print('Reading params form ' + filename)
        config = configparser.ConfigParser()
        config.read(filename)
        self.logdir = config.get('General', 'logdir')
        self.datapath = config.get('Train', 'data_file_path')

class DataReaderH5(T.DataHandler):

    def readH5(self, h5file):
        with h5py.File(h5file, 'r') as f:
            self.x = f['x'].value
            self.y = f['y'].value

    def unfold_targets(self):
        y_shape_new = list(self.y.shape)
        y_shape_new[-1] = 2
        y_new = np.zeros(y_shape_new, dtype=np.float32)
        for i in tqdm(range(y_shape_new[0]), desc='Image'):
            for j in range(y_shape_new[1]):
                for t in range(y_shape_new[2]):
                    tmp = self.y[i, j, t, 0]
                    try:
                        y_new[i, j, t, int(tmp)] = 1
                    except IndexError:
                        print('sacramento!')


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

    def network_squeeze_with_average(self, x, y):
        with tf.name_scope('Inputs'):
            input_tensor = x
            labels_tensor = y
            print('Input tensor {}'.format(input_tensor.get_shape()))
            print('labels tensor {}'.format(labels_tensor.get_shape()))
        with tf.name_scope('Encoder_1'):
            net = L.conv(input_tensor, name='conv1_1', kh=3, kw=3, n_out=40)
            net = L.conv(net, name='conv1_2', kh=3, kw=3, n_out=80)
            net = L.pool(net, name='pool1', kh=2, kw=2, dw=2, dh=2)
            print('Encoder1 {}'.format(net.get_shape()))
        with tf.name_scope('Encoder_2'):
            net = L.conv(net, name='conv2_1', kh=3, kw=3, n_out=90)
            net = L.conv(net, name='conv2_2', kh=3, kw=3, n_out=100)
            net = L.pool(net, name='pool2', kh=2, kw=2, dw=2, dh=2)
            print('Encoder2 {}'.format(net.get_shape()))
        with tf.name_scope('Decoder_1'):
            net = L.deconv(net, name='deconv1_1', kh=3, kw=3, n_out=101)
            net = L.deconv(net, name='deconv1_2', kh=3, kw=3, n_out=91)
            net = L.upsample(net, name='unpool1')
            print('Decoder1 {}'.format(net.get_shape()))
        with tf.name_scope('Decoder_2'):
            net = L.deconv(net, name='deconv2_1', kh=3, kw=3, n_out=81)
            net = L.deconv(net, name='deconv2_2', kh=3, kw=3, n_out=41)
            net = L.upsample(net, name='unpool2')
            print('Decoder2 {}'.format(net.get_shape()))
        with tf.name_scope('Logits'):
            net = tf.reduce_mean(net, axis=3, keep_dims=True, name='AVG_channels')
            print('Logits {}'.format(net.get_shape()))
            self.logits = tf.nn.sigmoid(net, 'sigm1')

        # compute the loss
        with tf.name_scope('Cost'):
            self.cost = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.logits, labels_tensor))))
        with tf.name_scope('Optimizer'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

    def network_squeeze(self, x, y):
        print('Input tensor {}'.format(x.get_shape()))
        print('labels tensor {}'.format(y.get_shape()))
        with tf.name_scope('Encoder_1'):
            net = L.conv(x, name='conv1_1', kh=3, kw=3, n_out=32)
            net = L.conv(net, name='conv1_2', kh=3, kw=3, n_out=32)
            net = L.pool(net, name='pool1', kh=2, kw=2, dw=2, dh=2)
            print('Encoder1 {}'.format(net.get_shape()))
        with tf.name_scope('Encoder_2'):
            net = L.conv(net, name='conv2_1', kh=3, kw=3, n_out=64)
            net = L.conv(net, name='conv2_2', kh=3, kw=3, n_out=64)
            net = L.pool(net, name='pool2', kh=2, kw=2, dw=2, dh=2)
            print('Encoder2 {}'.format(net.get_shape()))
        with tf.name_scope('Decoder_1'):
            net = L.deconv(net, name='deconv1_1', kh=3, kw=3, n_out=64)
            net = L.deconv(net, name='deconv1_2', kh=3, kw=3, n_out=64)
            net = L.upsample(net)
            print('Decoder1 {}'.format(net.get_shape()))
        with tf.name_scope('Decoder_2'):
            net = L.deconv(net, name='deconv2_1', kh=3, kw=3, n_out=32)
            net = L.deconv(net, name='deconv2_2', kh=3, kw=3, n_out=32)
            net = L.upsample(net)
            print('Decoder2 {}'.format(net.get_shape()))
        with tf.name_scope('Outputs'):
            net = L.conv(net, name='conv3', kh=5, kw=5, n_out=1)
            # net = L.flatten(net)
            print('Logits {}'.format(net.get_shape()))
            self.logits = tf.nn.sigmoid(net, 'sigm1')

        # compute the loss
        with tf.name_scope('Cost'):
            self.cost = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.logits, y))))
        with tf.name_scope('Optimizer'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

    def main(self):
        data_reader = DataReaderH5()
        data_reader.readH5(self.sets.datapath)
        # data_reader.unfold_targets()
        print('db_list length: {}'.format(data_reader.x.shape[0]))

        # generate the network
        print('creating the net')
        input_tensor = tf.placeholder(tf.float32,
                                      shape=[None, data_reader.x.shape[1], data_reader.x.shape[2], data_reader.x.shape[3]],
                                      name='x_input')
        labels_tensor = tf.placeholder(tf.float32,
                                       shape=[None, data_reader.y.shape[1], data_reader.y.shape[2], data_reader.y.shape[3]],
                                       name='y_input')
        self.network_squeeze(input_tensor, labels_tensor)

        tf.summary.scalar('loss', self.cost)
        tf.summary.image('input', input_tensor, max_outputs=3)
        tf.summary.image('masks', labels_tensor, max_outputs=3)
        tf.summary.image('output', self.logits, max_outputs=3)
        # tf.summary.image('norm_probe', self.norm_probe, max_outputs=3)
        # tf.summary.histogram('filters_1', self.filters)
        merged_summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(logdir=self.sets.logdir, graph=tf.get_default_graph(), flush_secs=10)
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            print('Entering the session.')
            sess.run(init)
            # Create a random validation set from data_reader
            val_img, val_mask = data_reader.fetch_random_validation_set_from_training(b_size=self.batch_size)
            # Start cycling epochs
            myiter = 0
            for epoch in range(0, self.epochs):
                te = time.time()
                print('+*+* Starting Epoch {}'.format(data_reader.epoch))
                for it in range(0, len(data_reader.x), self.batch_size):
                    myiter += 1
                    # fetch batch and labels
                    batch_data, batch_target = data_reader.next_batch(self.batch_size)
                    # run optimization
                    _, cost = sess.run([self.optimizer, self.cost], feed_dict={input_tensor: batch_data,
                                                                               labels_tensor: batch_target})
                    # print('sample {}/{} cost: {}'.format(it, len(data_reader.x), cost))

                val_loss, summary = sess.run([self.cost, merged_summary_op],
                                             feed_dict={input_tensor: val_img,
                                                        labels_tensor: val_mask})
                # vals = sess.run([self.logits], feed_dict={input_tensor: val_img, labels_tensor: val_mask})
                # output = vals[0]
                # for i in range(0, len(output)):
                #     T.show_image(output[i], self.sets.logdir)
                # tf.summary.image('Output', self.logits)
                summary_writer.add_summary(summary, epoch)
                print('Accuracy for epoch {}: {}'.format(epoch, val_loss))
                print('Time for epoch: {}'.format(time.time() - te))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='list of possible arguments')
    parser.add_argument('-s', type=str, metavar='config', help="config file input", default='config.cfg')
    args = parser.parse_args()
    set = Settings(args.s)
    m = Main(set)
    m.main()
