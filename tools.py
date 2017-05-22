import os
import numpy as np
import configparser
import glob
from scipy import misc
import random as rand
from matplotlib import pyplot as plt
import cv2
import tensorflow as tf


class Settings:
    def __init__(self, filename):
        print('Reading params form ' + filename)
        config = configparser.ConfigParser()
        config.read(filename)
        self.folder_tampered = config.get('General', 'folder_tp')
        self.folder_authentic = config.get('General', 'folder_au')
        self.folder_mask = config.get('General', 'folder_mask')
        self.logdir = config.get('General', 'logdir')

        self.datapath = config.get('Train', 'data_file_path')

    def __str__(self):
        return 'tp: {}\nau: {}\nmask: {}'.format(self.folder_tampered,
                                                 self.folder_authentic,
                                                 self.folder_mask)

    def check_dataset(self):
        '''
        create the list of file
        :return: dataset --> list of tuple(file_path, mask_path)
        '''
        files_tp = glob.glob1(self.folder_tampered, '*')
        files_au = glob.glob1(self.folder_authentic, '*')
        files_mask = glob.glob1(self.folder_mask, '*')
        if len(files_mask) != len(files_tp):
            print('n_tp: {}\nn_au: {}\nn_mask: {}\n'.format(len(files_tp), len(files_au), len(files_mask)))
            raise Exception('Files mismatch between tp and mask')
        dataset = []
        for i, name in enumerate(files_tp):
            tmp_file = os.path.join(self.folder_tampered, name)
            mask_file = glob.glob1(self.folder_mask, '{}*'.format(name.split('.')[:-1][0]))
            if len(mask_file) == 0:
                raise Exception('bad news! the mask for {} is missing!'.format(name))
            dataset.append((tmp_file, os.path.join(self.folder_mask, mask_file[0])))
        return dataset


class DataHandler:
    def __init__(self):
        self.b_pointer = 0
        self.epoch = 1
        self.x = None
        self.y = None

    def reset_training(self):
        self.b_pointer = 0
        self.epoch = 1

    def shuffle_db(self):
        n_imgs = self.x.shape[0]
        ord = [i for i in range(n_imgs)]
        rand.shuffle(ord)
        self.x = self.x[ord, :, :, :]
        self.y = self.y[ord, :, :, :]
        return ord

    def next_batch(self, b_size):
        b_idx = [self.b_pointer, self.b_pointer + b_size]
        # check if the batch is finished or returning 0
        self.b_pointer += b_size
        batch_imgs = self.x[b_idx[0]:b_idx[1], :, :, :]
        batch_masks = self.y[b_idx[0]:b_idx[1], :, :, :]
        if len(batch_imgs) < b_size:
            self.shuffle_db()
            self.b_pointer = 0
            self.epoch += 1
        return batch_imgs, batch_masks

    def fetch_random_validation_set_from_training(self, b_size):
        r = rand.randint(0, len(self.x) - b_size)
        b_idx = [r, r + b_size]
        batch_imgs = self.x[b_idx[0]:b_idx[1], :, :, :]
        batch_masks = self.y[b_idx[0]:b_idx[1], :, :, :]
        return batch_imgs, batch_masks


def show_image(img, where=None):
    print('img_shape: {}'.format(img.shape))
    import scipy.misc as ms
    if len(img.shape) == 3:
        if img.shape[2] == 1:
            img = img[:, :, 0]
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.title('Image Plot')
    plt.show()
    if where is None:
        ms.imsave('to_be_deleted_if_found_in_the_hard_disk.jpg', img)
    else:
        path = os.path.join(where, 'to_be_deleted_if_found_in_the_hard_disk.jpg')
        ms.imsave(path, img)

