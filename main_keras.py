import sys
import os
import argparse
import numpy as np
import tensorflow as tf
import argparse
import configparser
import glob
from scipy import misc
import matplotlib.pyplot as plt
import tqdm
import time
import layers as L
import random as rand
import tools as T
from keras.optimizers import SGD, Adam
import keras.backend as K
from keras.models import Sequential, Model
from keras.models import Input
from keras.layers import concatenate, Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Deconv2D, UpSampling2D
from keras.callbacks import TensorBoard
import scipy.misc as ms

class Main:
    def __init__(self, sets):
        self.sets = sets
        self.size_image = 64
        self.n_channels = 3
        self.batch_size = 32
        self.learning_rate = 0.001
        self.epochs = 500

        # net parameters
        self.logits = None
        self.cost = None
        # self.accuracy = None
        self.optimizer = None

    def network_squeeze(self, x_shape):
        model = Sequential()
        model.add(
            Convolution2D(64, (3, 3), padding='same',
                          input_shape=x_shape))
        model.add(Activation('relu'))
        model.add(
            Convolution2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(
            Convolution2D(128, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(
            Convolution2D(128, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        print('output dopo 2 pooling: {}'.format(model.output_shape))

        model.add(Deconv2D(128, (3, 3), padding='same'))
        model.add(Deconv2D(128, (3, 3), padding='same'))
        print('output dopo 1 deconv: {}'.format(model.output_shape))
        model.add(UpSampling2D(size=(2, 2)))
        print('output dopo 1 upsampling: {}'.format(model.output_shape))

        model.add(Deconv2D(64, (3, 3), padding='same'))
        model.add(Deconv2D(64, (3, 3), padding='same'))
        print('output dopo 2 deconv: {}'.format(model.output_shape))
        model.add(UpSampling2D(size=(2, 2)))
        print('output dopo 2 upsampling: {}'.format(model.output_shape))

        model.add(Lambda(lambda l: K.mean(l, axis=3, keepdims=True)))
        model.add(Activation('sigmoid'))
        return model

    def network_concat(self, x_shape):
        # myinput = Input(shape=(x_shape[1], x_shape[2], x_shape[3]), name='input')
        myinput = Input(shape=x_shape, name='input')
        c1 = Conv2D(filters=32,
                    kernel_size=(3, 3),
                    padding='same',
                    activation='sigmoid')(myinput)
        # c1 = Flatten()(c1)
        c2 = Conv2D(filters=32,
                    kernel_size=(5, 5),
                    padding='same',
                    activation='sigmoid')(myinput)
        # c2 = Flatten()(c2)
        c3 = Conv2D(filters=32,
                    kernel_size=(7, 7),
                    padding='same',
                    activation='sigmoid')(myinput)
        # c3 = Flatten()(c3)
        c4 = Conv2D(filters=32,
                    kernel_size=(9, 9),
                    padding='same',
                    activation='sigmoid')(myinput)
        # c4 = Flatten()(c4)
        m1 = concatenate([c1, c2, c3, c4])
        c5 = Conv2D(filters=1,
                    kernel_size=(5, 5),
                    padding='same',
                    activation='sigmoid')(m1)
        model = Model(inputs=myinput, outputs=c5)
        model.summary()
        return model

    def tmpfunc(self, x_shape):
        input_img = Input(shape=x_shape)

        tower_1 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_img)
        tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower_1)

        tower_2 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_img)
        tower_2 = Conv2D(64, (5, 5), padding='same', activation='relu')(tower_2)

        tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_img)
        tower_3 = Conv2D(64, (1, 1), padding='same', activation='relu')(tower_3)

        output = concatenate([tower_1, tower_2, tower_3], axis=1)
        model = Model(inputs=input_img, outputs=output)
        model.summary()
        return model

    def main(self):
        db_list = self.sets.check_dataset()

        print('db_list length: {}'.format(len(db_list)))
        # model = self.network_squeeze((None, self.size_image, self.size_image, 3))
        model = self.network_squeeze((self.size_image, self.size_image, 3))
        data_reader = T.DataReader()
        data_reader.read_data(db_list)
        data_reader.prepare_data_patches(self.size_image, self.size_image)

        opt = Adam(lr=self.learning_rate)
        model.compile(loss='mean_squared_error', optimizer=opt)

        model.fit(data_reader.images, data_reader.masks, batch_size=self.batch_size, epochs=self.epochs,
                  callbacks=[TensorBoard(log_dir=set.logdir, histogram_freq=0, write_graph=True, write_images=False)])

        val_img, val_mask = data_reader.fetch_random_validation_set_from_training(b_size=self.batch_size)
        myeval = model.evaluate(val_img, val_mask, batch_size=self.batch_size)
        classes = model.predict(val_img, batch_size=self.batch_size)
        for i in range(len(classes)):
            img1 = classes[i][:, :, 0]
            img1 = np.round(img1 * 255)
            maskout = os.path.join(self.sets.logdir, 'out_{}_output.jpg'.format(i))
            imgout = os.path.join(self.sets.logdir, 'out_{}_img.jpg'.format(i))
            labkout = os.path.join(self.sets.logdir, 'out_{}_mask.jpg'.format(i))
            ms.imsave(labkout, val_mask[i, :, :, 0])
            ms.imsave(imgout, val_img[i])
            ms.imsave(maskout, img1)

        print('final eval : {}'.format(myeval))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='list of possible arguments')
    parser.add_argument('-s', type=str, metavar='config', help="config file input", default='config.cfg')
    args = parser.parse_args()
    set = T.Settings(args.s)
    m = Main(set)
    print(m.sets)
    m.main()
