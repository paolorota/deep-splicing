import os
import numpy as np
import tensorflow as tf
import configparser
import datetime
from scipy import misc
import matplotlib.pyplot as plt
import h5py
import time
from layers import *
from tqdm import tqdm
import random as rand
from six.moves import xrange
import scipy
from scipy.fftpack import dct

def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


class Settings:
    def __init__(self, filename):
        print('Reading params form ' + filename)
        config = configparser.ConfigParser()
        config.read(filename)
        self.logdir = config.get('General', 'logdir')
        self.datapath = config.get('Train', 'data_file_path')
        self.to_gray = bool(int(config.get('Train', 'transform_to_gray')))


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

    def to_gray(self):
        x_new = np.zeros_like(self.y)
        for i in range(x_new.shape[0]):
            tmp_x = np.reshape(self.x[i], (self.x.shape[1], self.x.shape[2], self.x.shape[3]))
            x_new[i, :, :, 0] = np.dot(tmp_x, [0.299, 0.587, 0.114])
        self.x = x_new

    def to_dct(self):
        self.to_gray()
        for i in range(self.x.shape[0]):
            for y in range(0, self.x.shape[1], 8):
                for x in range(0, self.x.shape[2], 8):
                    tmp = self.x[i, y:y+8, x:x+8, 0]
                    tmp = dct(dct(tmp, axis=0), axis=1)
                    tmp[0, 0] = 0
                    self.x[i, y:y + 8, x:x + 8, 0] = tmp



class Main:
    def __init__(self, sets):
        self.sets = sets
        self.size_image = 128
        self.batch_size = 32
        self.learning_rate = 0.0001
        self.epochs = 400
        self.sample_num = 32
        if sets.to_gray:
            self.n_channels = 1
        else:
            self.n_channels = 3

        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')
        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')
        self.g_bn4 = batch_norm(name='g_bn4')

        self.gf_dim = 64
        self.gfc_dim = 1024
        self.c_dim = 1  # mask is one channel
        self.df_dim = 64

        self.z_dim = 100

    def generator(self, x):
        with tf.variable_scope("generator") as scope:
            if self.n_channels == 3:
                # to 128x128x3 --> 128x128x1
                h0_ = lrelu(conv2d(x, 1, d_h=1, d_w=1, name='g_hA_conv'))
                s_h0_ = h0_.get_shape()
            else:
                s_h0_ = x.get_shape()
            # down to 64x64x64
            h0 = lrelu(conv2d(x, self.gf_dim, name='g_h0_conv'))
            s_h0 = h0.get_shape()
            # down to 32x32x128
            h1 = lrelu(conv2d(h0, self.gf_dim * 2, name='g_h1_conv'))
            s_h1 = h1.get_shape()
            # down to 16x16x256
            h2 = lrelu(conv2d(h1, self.gf_dim * 4, name='g_h2_conv'))
            s_h2 = h2.get_shape()
            # down to 8x8x512
            h3 = lrelu(conv2d(h2, self.gf_dim * 8, name='g_h3_conv'))
            s_h3 = h3.get_shape()

            # up to 16x16x256
            h4, h4_w, h4_b = deconv2d(
                h3, [self.batch_size, s_h2[1].value, s_h2[2].value, self.gf_dim * 4], name='g_h4', with_w=True)
            h4 = tf.nn.relu(self.g_bn4(h4))
            # merge h4 and h2 --> 16x16x512
            h5 = concat([h4, h2], 3)
            # up to 32x32x128
            h5, h5_w, h5_b = deconv2d(
                h5, [self.batch_size, s_h1[1].value, s_h1[2].value, self.gf_dim * 2], name='g_h5', with_w=True)
            h5 = tf.nn.relu(self.g_bn3(h5))
            # merge h5 and h1 --> 32x32x256
            h6 = concat([h5, h1], 3)
            # up to 64x64x128
            h6, h6_w, h6_b = deconv2d(
                h6, [self.batch_size, s_h0[1].value, s_h0[2].value, self.gf_dim], name='g_h6', with_w=True)
            h6 = tf.nn.relu(self.g_bn2(h6))
            # merge h6 and h0 --> 64x64x128
            h7 = concat([h6, h0], 3)
            # up to 128x128x1
            h7, h7_w, h7_b = deconv2d(
                h7, [self.batch_size, s_h0_[1].value, s_h0_[2].value, self.c_dim], name='g_h7', with_w=True)
            return tf.nn.sigmoid(h7)

    def sampler(self, x):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()
            if self.n_channels == 3:
                # to 128x128x3 --> 128x128x1
                h0_ = lrelu(conv2d(x, 1, d_h=1, d_w=1, name='g_hA_conv'))
                s_h0_ = h0_.get_shape()
            else:
                s_h0_ = x.get_shape()
            # down to 64x64x64
            h0 = lrelu(conv2d(x, self.gf_dim, name='g_h0_conv'))
            s_h0 = h0.get_shape()
            # down to 32x32x128
            h1 = lrelu(conv2d(h0, self.gf_dim * 2, name='g_h1_conv'))
            s_h1 = h1.get_shape()
            # down to 16x16x256
            h2 = lrelu(conv2d(h1, self.gf_dim * 4, name='g_h2_conv'))
            s_h2 = h2.get_shape()
            # down to 8x8x512
            h3 = lrelu(conv2d(h2, self.gf_dim * 8, name='g_h3_conv'))
            s_h3 = h3.get_shape()

            # up to 16x16x256
            h4, h4_w, h4_b = deconv2d(
                h3, [self.batch_size, s_h2[1].value, s_h2[2].value, self.gf_dim * 4], name='g_h4', with_w=True)
            h4 = tf.nn.relu(self.g_bn4(h4))
            # merge h4 and h2 --> 16x16x512
            h5 = concat([h4, h2], 3)
            # up to 32x32x128
            h5, h5_w, h5_b = deconv2d(
                h5, [self.batch_size, s_h1[1].value, s_h1[2].value, self.gf_dim * 2], name='g_h5', with_w=True)
            h5 = tf.nn.relu(self.g_bn3(h5))
            # merge h5 and h1 --> 32x32x256
            h6 = concat([h5, h1], 3)
            # up to 64x64x128
            h6, h6_w, h6_b = deconv2d(
                h6, [self.batch_size, s_h0[1].value, s_h0[2].value, self.gf_dim], name='g_h6', with_w=True)
            h6 = tf.nn.relu(self.g_bn2(h6))
            # merge h6 and h0 --> 64x64x128
            h7 = concat([h6, h0], 3)
            # up to 128x128x1
            h7, h7_w, h7_b = deconv2d(
                h7, [self.batch_size, s_h0_[1].value, s_h0_[2].value, self.c_dim], name='g_h7', with_w=True)
            return tf.nn.sigmoid(h7)

    def siamese_tower(self, image, reuse=False):
        with tf.variable_scope("siamese_tower") as scope:
            if reuse:
                scope.reuse_variables()
            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim * 2, name='d_h1_conv')))
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim * 4, name='d_h2_conv')))
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim * 8, name='d_h3_conv')))
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 128, 'd_h3_lin')
            return h4

    def siamese_discriminator(self, image, mask, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            if self.n_channels == 3:
                t0 = lrelu(conv2d(image, 1, d_h=1, d_w=1, name='d_hA_conv'))
            else:
                t0 = image
            t1 = self.siamese_tower(t0, reuse=reuse)
            t2 = self.siamese_tower(mask, reuse=True)
            t3 = concat([t1, t2], 1)
            t4 = linear(t3, 1, 'd_h4_lin')
            return tf.nn.sigmoid(t4), t4

    def build_gan(self):
        print('creating the net')
        self.x = tf.placeholder(tf.float32,
                                shape=[self.batch_size, self.size_image, self.size_image, self.n_channels],
                                name='img_input')
        self.y = tf.placeholder(tf.float32,
                                shape=[self.batch_size, self.size_image, self.size_image, 1],
                                name='mask_input')
        # self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z_input')
        # TODO: maybe use the learning rate!
        self.lr = tf.placeholder(tf.float32, name='learning_rate')

        self.x_sum = tf.summary.image("x", self.x)
        self.y_sum = tf.summary.image("y", self.y)

        self.G = self.generator(self.x)
        self.D, self.D_logits = self.siamese_discriminator(self.x, self.y, reuse=False)
        self.D_, self.D_logits_ = self.siamese_discriminator(self.x, self.G, reuse=True)
        self.S = self.sampler(self.x)

        # print('Noise tensor {}'.format(self.z.get_shape()))
        print('Image tensor {}'.format(self.x.get_shape()))
        print('Mask tensor {}'.format(self.y.get_shape()))
        self.d_sum = tf.summary.histogram("d", self.D)
        self.d__sum = tf.summary.histogram("d_", self.D_)
        self.G_sum = tf.summary.image("G", self.G)


        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_)))

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
            self.g_loss_sum, self.G_sum, self.d__sum
        ])
        self.d_sum = tf.summary.merge([
            self.d_loss_sum, self.d_sum
        ])
        self.in_sum = tf.summary.merge([self.x_sum, self.y_sum])
        self.writer = tf.summary.FileWriter(self.log_dir, sess.graph)

        # sampling random noise
        # sample_z = np.random.uniform(-1, 1, size=(self.sample_num, self.z_dim))

        # sample inputs and labels
        sample_inputs = self.data_reader.x[0:self.sample_num]
        sample_labels = self.data_reader.y[0:self.sample_num]

        counter = 1
        for epoch in xrange(self.epochs):
            # print('Starting epoch: {}'.format(epoch + 1))
            batch_idxs = len(self.data_reader.x) // self.batch_size

            for idx in xrange(0, batch_idxs):
                batch_images = self.data_reader.x[idx*self.batch_size:(idx+1)*self.batch_size]
                batch_labels = self.data_reader.y[idx*self.batch_size:(idx+1)*self.batch_size]
                # batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

                # Update D network
                _, summary_str, summary_str1 = sess.run([d_optim, self.d_sum, self.in_sum],
                                                        feed_dict={
                                                            self.x: batch_images,
                                                            self.y: batch_labels
                                                        })
                # step = (np.float32(epoch) + 1)+(np.float32(idx)/np.float32(batch_idxs))
                self.writer.add_summary(summary=summary_str, global_step=counter)
                self.writer.add_summary(summary=summary_str1, global_step=counter)

                # Update G (twice because it is suggested like so)
                for j in xrange(2):
                    _, summary_str = sess.run([g_optim, self.g_sum],
                                              feed_dict={
                                                  self.x: batch_images,
                                                  self.y: batch_labels
                                              })
                self.writer.add_summary(summary=summary_str, global_step=counter)

                # compute error on training
                errD_fake = self.d_loss_fake.eval({
                    self.x: batch_images,
                    self.y: batch_labels
                })
                errD_real = self.d_loss_real.eval({
                    self.x: batch_images,
                    self.y: batch_labels
                })
                errG = self.g_loss.eval({
                    self.x: batch_images,
                    self.y: batch_labels
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
                            self.x: sample_inputs,
                            self.y: sample_labels
                        }
                    )
                    manifold_h = int(np.ceil(np.sqrt(samples.shape[0])))
                    manifold_w = int(np.ceil(np.sqrt(samples.shape[0])))
                    save_images(samples, [manifold_h, manifold_w], './{}/train_{:02d}_{:04d}_rec.png'.format(
                        self.sample_dir, epoch, idx))
                    save_images(sample_labels, [manifold_h, manifold_w], './{}/train_{:02d}_{:04d}_mask.png'.format(
                        self.sample_dir, epoch, idx))
                    save_images(sample_inputs, [manifold_h, manifold_w], './{}/train_{:02d}_{:04d}_real.png'.format(
                        self.sample_dir, epoch, idx))
                    print("[Sample] d_loss: {0:.8}, g_loss: {1:.8}".format(d_loss, g_loss))
                counter += 1

    def main(self):
        launch_time = datetime.datetime.now()
        log = 'fgan_y{0:4}_m{1:2}_d{2:2}_h{3:2}{4:2}{5:2}'.format(launch_time.year,
                                                                  launch_time.month,
                                                                  launch_time.day,
                                                                  launch_time.hour,
                                                                  launch_time.minute,
                                                                  launch_time.second).replace(' ', '0')
        self.log_dir = os.path.join(self.sets.logdir, log)
        if not os.path.exists(self.sets.logdir):
            os.mkdir(self.sets.logdir)
        self.sample_dir = os.path.join('samples', log)
        if not os.path.exists(self.sample_dir):
            os.mkdir(self.sample_dir)
        self.data_reader = DataReaderH5(self.sets.datapath)
        #optional
        if self.n_channels == 1:
            # print('Performing DCT 8x8 on self.x')
            # self.data_reader.to_dct()
            self.data_reader.to_dct()

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
