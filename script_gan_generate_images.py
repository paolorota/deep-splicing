import os
import numpy as np
import tensorflow as tf
import configparser
import datetime
from scipy import misc
import matplotlib.pyplot as plt
import h5py
import time
from layers2 import *
from tqdm import tqdm
import random as rand
from six.moves import xrange
import scipy

def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


class Settings:
    def __init__(self, filename):
        print('Reading params form ' + filename)
        config = configparser.ConfigParser()
        config.read(filename)
        self.logdir = config.get('General', 'logdir')
        self.datapath = config.get('Train', 'data_file_path')


class DataReaderH5:
    def __init__(self, h5file):
        self.x = None
        self.y = None
        self.readH5(h5file)

    def shuffle_db(self):
        n_imgs = self.x.shape[0]
        ord = [i for i in range(n_imgs)]
        rand.shuffle(ord)
        self.x = self.x[ord, :, :, :]
        self.y = self.y[ord, :, :, :]
        return ord

    def readH5(self, h5file):
        with h5py.File(h5file, 'r') as f:
            self.x = f['x'].value
            self.y = f['y'].value

class Main:
    def __init__(self, sets):
        self.sets = sets
        self.size_image = 128
        self.n_channels = 3
        self.batch_size = 32
        self.learning_rate = 0.0001
        self.epochs = 5000
        self.sample_num = 32

        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')
        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')

        self.gf_dim = 64
        self.gfc_dim = 1024
        self.c_dim = 3  # mask is one channel
        self.df_dim = 64

        self.z_dim = 100

    def generator(self, z):
        with tf.variable_scope("generator") as scope:
            s_h, s_w = self.size_image, self.size_image
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

            # project `z` and reshape
            self.z_, self.h0_w, self.h0_b = linear(
                z, self.gf_dim * 8 * s_h16 * s_w16, 'g_h0_lin', with_w=True)

            self.h0 = tf.reshape(
                self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(self.h0))

            self.h1, self.h1_w, self.h1_b = deconv2d(
                h0, [self.batch_size, s_h8, s_w8, self.gf_dim * 4], name='g_h1', with_w=True)
            h1 = tf.nn.relu(self.g_bn1(self.h1))

            h2, self.h2_w, self.h2_b = deconv2d(
                h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_h2', with_w=True)
            h2 = tf.nn.relu(self.g_bn2(h2))

            h3, self.h3_w, self.h3_b = deconv2d(
                h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_h3', with_w=True)
            h3 = tf.nn.relu(self.g_bn3(h3))

            h4, self.h4_w, self.h4_b = deconv2d(
                h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)
            return tf.nn.sigmoid(h4)

    def discriminator(self, image, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim * 2, name='d_h1_conv')))
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim * 4, name='d_h2_conv')))
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim * 8, name='d_h3_conv')))
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')
            return tf.nn.sigmoid(h4), h4

    def sampler(self, z):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()

            s_h, s_w = self.size_image, self.size_image
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

            # project `z` and reshape
            h0 = tf.reshape(
                linear(z, self.gf_dim * 8 * s_h16 * s_w16, 'g_h0_lin'),
                [-1, s_h16, s_w16, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(h0, train=False))
            h1 = deconv2d(h0, [self.sample_num, s_h8, s_w8, self.gf_dim * 4], name='g_h1')
            h1 = tf.nn.relu(self.g_bn1(h1, train=False))
            h2 = deconv2d(h1, [self.sample_num, s_h4, s_w4, self.gf_dim * 2], name='g_h2')
            h2 = tf.nn.relu(self.g_bn2(h2, train=False))
            h3 = deconv2d(h2, [self.sample_num, s_h2, s_w2, self.gf_dim * 1], name='g_h3')
            h3 = tf.nn.relu(self.g_bn3(h3, train=False))
            h4 = deconv2d(h3, [self.sample_num, s_h, s_w, self.c_dim], name='g_h4')
            return tf.nn.sigmoid(h4)

    def build_gan(self):
        print('creating the net')
        self.x = tf.placeholder(tf.float32,
                                shape=[self.batch_size, self.size_image, self.size_image, self.n_channels],
                                name='x_input')
        # self.s = tf.placeholder(tf.float32,
        #                         shape=[self.sample_num, self.size_image, self.size_image, self.n_channels],
        #                         name='x_input')
        ## Al momento non utilizzo la mask
        # self.y = tf.placeholder(tf.float32,
        #                         shape=[self.batch_size, self.size_image, self.size_image, 1],
        #                         name='x_input')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z_input')
        # TODO: maybe use the learning rate!
        self.lr = tf.placeholder(tf.float32, name='learning_rate')

        self.G = self.generator(self.z)
        self.D, self.D_logits = self.discriminator(self.x, reuse=False)
        self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)
        # TODO: do we need a sampler?? if yes, implement it!
        self.S = self.sampler(self.z)

        print('Noise tensor {}'.format(self.z.get_shape()))
        print('Image tensor {}'.format(self.x.get_shape()))
        # print('Mask tensor {}'.format(self.y.get_shape()))

        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, targets=tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, targets=tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, targets=tf.ones_like(self.D_)))

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = tf.summary.scalar('g_loss', self.g_loss)
        self.d_loss_sum = tf.summary.scalar('d_loss', self.d_loss)

        # select the variables to train and where
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

    def train_gan(self, sess):
        d_optim = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(self.g_loss, var_list=self.g_vars)
        tf.global_variables_initializer().run()

        # put together summaries
        self.g_sum = tf.summary.merge([
            self.g_loss_sum
        ])
        self.d_sum = tf.summary.merge([
            self.d_loss_sum
        ])
        self.writer = tf.summary.FileWriter(self.log_dir, sess.graph)

        # sampling random noise
        sample_z = np.random.uniform(-1, 1, size=(self.sample_num, self.z_dim))

        # sample inputs and labels
        sample_inputs = self.data_reader.x[0:self.sample_num]
        # sample_labels = self.data_reader.y[0:self.sample_num]

        counter = 1
        for epoch in xrange(self.epochs):
            # print('Starting epoch: {}'.format(epoch + 1))
            batch_idxs = len(self.data_reader.x) // self.batch_size

            for idx in xrange(0, batch_idxs):
                batch_images = self.data_reader.x[idx*self.batch_size:(idx+1)*self.batch_size]
                # batch_labels = self.data_reader.y[idx*self.batch_size:(idx+1)*self.batch_size]
                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

                # Update D network
                _, summary_str = sess.run([d_optim, self.d_sum],
                                          feed_dict={
                                              self.x: batch_images,
                                              self.z: batch_z
                                          })
                # step = (np.float32(epoch) + 1)+(np.float32(idx)/np.float32(batch_idxs))
                self.writer.add_summary(summary=summary_str, global_step=counter)

                # Update G (twice because it is suggested like so)
                for j in xrange(2):
                    _, summary_str = sess.run([g_optim, self.g_sum],
                                              feed_dict={
                                                  self.z: batch_z
                                              })
                self.writer.add_summary(summary=summary_str, global_step=counter)

                # compute error on training
                errD_fake = self.d_loss_fake.eval({
                    self.x: batch_images,
                    self.z: batch_z
                })
                errD_real = self.d_loss_real.eval({
                    self.x: batch_images,
                    self.z: batch_z
                })
                errG = self.g_loss.eval({
                    self.z: batch_z
                })
                # counter += 1
                print('Epoch [{0}] [{1}/{2}] d_loss: {3:.8} g_loss: {4:.8}'.format(
                    epoch, idx, batch_idxs, errD_fake+errD_real, errG
                ))

                # save images every 100 batch iters
                if np.mod(counter, 100) == 1:
                    samples, d_loss, g_loss = sess.run(
                        [self.S, self.d_loss, self.g_loss],
                        feed_dict={
                            self.z: sample_z,
                            self.x: sample_inputs
                        }
                    )
                    manifold_h = int(np.ceil(np.sqrt(samples.shape[0])))
                    manifold_w = int(np.ceil(np.sqrt(samples.shape[0])))
                    save_images(samples, [manifold_h, manifold_w], './{}/train_{:02d}_{:04d}.png'.format(
                        self.sample_dir, epoch, idx))
                    print("[Sample] d_loss: {0:.8}, g_loss: {1:.8}".format(d_loss, g_loss))
                counter += 1

    def main(self):
        launch_time = datetime.datetime.now()
        log = 'fgan_{0:4}{1:2}{2:2}_{3:2}{4:2}{5:2}'.format(launch_time.year,
                                                            launch_time.month,
                                                            launch_time.day,
                                                            launch_time.hour,
                                                            launch_time.minute,
                                                            launch_time.second).replace(' ', '0')
        self.log_dir = os.path.join(self.sets.logdir, log)
        if not os.path.exists(self.sets.logdir):
            os.mkdir(self.sets.logdir)
        self.sample_dir = os.path.join('./samples', log)
        if not os.path.exists(self.sample_dir):
            os.mkdir(self.sample_dir)
        self.data_reader = DataReaderH5(self.sets.datapath)

        print('db_list length: {}'.format(self.data_reader.x.shape[0]))

        # generate the network
        self.build_gan()

        with tf.Session() as sess:
            self.train_gan(sess=sess)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if images.shape[3] in (3,4):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3] == 1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter '
                         'must have dimensions: HxW or HxWx3 or HxWx4')


def save_images(images, size, image_path):
    return imsave(images, size, image_path)


def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='list of possible arguments')
    parser.add_argument('-s', type=str, metavar='config', help="config file input", default='config.cfg')
    args = parser.parse_args()
    set = Settings(args.s)
    m = Main(set)
    m.main()
