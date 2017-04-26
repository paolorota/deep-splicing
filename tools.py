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


class DataReader:
    def __init__(self):
        self.textual_db = None
        self.image_db = None
        self.b_pointer = 0
        self.epoch = 1
        self.images = None
        self.masks = None

    def read_data(self, db):
        self.textual_db = db
        self.image_db = []
        for i, n in enumerate(db):
            im1 = misc.imread(n[0], mode='RGB')
            im2 = misc.imread(n[1], flatten=True)
            im3 = cv2.imread(n[0])
            # b, g, r = cv2.split(im3)
            # rgb_img = cv2.merge([r, g, b])
            self.image_db.append((im1, im2))
            # self.n_imgs = len(self.image_db)

    def prepare_data_patches(self, iw, ih, pximage=10):
        db_size = len(self.image_db)
        im_tensor = np.zeros((db_size*pximage, iw, ih, 3), dtype=np.float32)
        mk_tensor = np.zeros((db_size*pximage, iw, ih, 1), dtype=np.float32)
        im_names = list()
        myiter = 0
        for i, n in enumerate(self.image_db):
            img = n[0] / 255
            mask = n[1] / 255
            image_shape = img.shape[0:-1]
            mask_shape = mask.shape
            # for unknown reasons some masks are not of the same size of the images
            if image_shape != mask_shape:
                mask = misc.imresize(mask, image_shape)
                print('This image does not work {} - Performing resize of the mask'.format(i))
            limitH = (0, image_shape[0] - ih)
            limitW = (0, image_shape[1] - iw)
            tmp_check = 1
            while True:
                h0 = rand.randint(limitH[0], limitH[1])
                w0 = rand.randint(limitW[0], limitW[1])
                im1 = img[h0:h0+ih, w0:w0+iw, :]
                im2 = mask[h0:h0+ih, w0:w0+iw]
                # condition for having some tampering region into the patch
                if np.sum(im2) == 0 or np.sum(im2) == iw * ih:
                    tmp_check += 1
                    if tmp_check % 200 == 0:
                        print('tmp_check for image {}/{} = {}'.format(i, len(self.image_db), tmp_check))
                    continue
                # plt.subplot(221)
                # plt.imshow(img)
                # plt.xticks([])
                # plt.yticks([])
                # plt.title('Image tampered')
                # plt.subplot(222)
                # plt.imshow(mask)
                # plt.xticks([])
                # plt.yticks([])
                # plt.title('Image mask')
                # plt.subplot(223)
                # plt.imshow(im1)
                # plt.xticks([])
                # plt.yticks([])
                # plt.title('Patch tampered')
                # plt.subplot(224)
                # plt.imshow(im2)
                # plt.xticks([])
                # plt.yticks([])
                # plt.title('Patch mask')
                # plt.show()
                im_tensor[myiter] = im1
                mk_tensor[myiter, :, :, 0] = im2
                im_names.append(self.textual_db[i])
                myiter += 1
                if myiter % pximage == 0:
                    break
        self.textual_db = im_names
        self.images = im_tensor
        self.masks = mk_tensor

    def prepare_data_resize(self, iw, ih):
        db_size = len(self.image_db)
        im_tensor = np.zeros((db_size, iw, ih, 3), dtype=np.float32)
        mk_tensor = np.zeros((db_size, iw, ih, 1), dtype=np.float32)
        for i, n in enumerate(self.image_db):
            im1 = misc.imresize(n[0], (iw, ih))
            im2 = misc.imresize(n[1], (iw, ih))
            im2 = im2 / 255
            im_tensor[i, :, :, :] = im1[:, :, 0:3]
            if len(im2.shape) > 2:
                mk_tensor[i, :, :, 0] = im2[:, :, 0]
                raise Exception('This condition should never happens')
            else:
                mk_tensor[i, :, :, 0] = im2[:, :]
            # plt.subplot(221)
            # plt.imshow(im1)
            # plt.xticks([])
            # plt.yticks([])
            # plt.title('Image tampered')
            # plt.subplot(222)
            # plt.imshow(im2, cmap='gray')
            # plt.xticks([])
            # plt.yticks([])
            # plt.title('Image mask')
            # plt.subplot(223)
            # plt.imshow(im_tensor[i, :, :, :])
            # plt.xticks([])
            # plt.yticks([])
            # plt.title('Image tensor')
            # plt.subplot(224)
            # tmp = im_tensor[i, :, :, :]
            # tmp = tmp.astype(np.uint8)
            # plt.imshow(tmp)
            # plt.xticks([])
            # plt.yticks([])
            # plt.title('Image tensor int')
            # plt.show()
        self.images = im_tensor
        self.masks = mk_tensor



    def reset_training(self):
        self.b_pointer = 0
        self.epoch = 1

    def shuffle_db(self):
        n_imgs = len(self.images)
        ord = [i for i in range(n_imgs)]
        rand.shuffle(ord)
        self.images = self.images[ord, :, :, :]
        self.masks = self.masks[ord, :, :, :]
        tmp = [self.textual_db[i] for i in ord]
        self.textual_db = tmp

    def next_batch(self, b_size):
        b_idx = [self.b_pointer, self.b_pointer + b_size]
        # check if the batch is finished or returning 0
        self.b_pointer += b_size
        batch_imgs = self.images[b_idx[0]:b_idx[1], :, :, :]
        batch_masks = self.masks[b_idx[0]:b_idx[1], :, :, :]
        if len(batch_imgs) < b_size:
            self.shuffle_db()
            self.b_pointer = 0
            self.epoch += 1
        return batch_imgs, batch_masks

    def fetch_random_validation_set_from_training(self, b_size):
        r = rand.randint(0, len(self.images) - b_size)
        b_idx = [r, r + b_size]
        batch_imgs = self.images[b_idx[0]:b_idx[1], :, :, :]
        batch_masks = self.masks[b_idx[0]:b_idx[1], :, :, :]
        return batch_imgs, batch_masks

    def fetch_and_show(self, i):
        tmp_img = self.images[i, :, :, :]
        tmp_mask = self.masks[i, :, :, :]
        plt.subplot(121)
        plt.imshow(tmp_img)
        plt.subplot(122)
        plt.imshow(tmp_mask.reshape((tmp_mask.shape[:-1])), cmap='gray')
        plt.suptitle(self.textual_db[i][0])
        plt.show()


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

