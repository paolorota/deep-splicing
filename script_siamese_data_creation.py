import os
import configparser
import glob
from scipy import misc
import numpy as np
import cv2
import random as rand
from time import time
import datetime

import h5py


class DataManager_siamese:
    def __init__(self):
        self.b_pointer = 0
        self.epoch = 1
        self.x1 = None
        self.x2 = None
        self.y = None
        self.textual_db = None
        self.image_db = None

    def read_data(self, db):
        self.textual_db = db
        self.image_db = []
        for i, n in enumerate(db):
            im1 = misc.imread(n[0], mode='RGB')
            im2 = misc.imread(n[1], flatten=True)
            self.image_db.append((im1, im2))

    def _cycle_for_patches(self, iw, ih, p1val, p2val, label, pximage=10):
        im1_tensor = list()
        im2_tensor = list()
        lb_tensor = list()
        im_names = list()
        # one white / one black
        print('one white / one black')
        for i, n in enumerate(self.image_db):
            print('{}/{}'.format(i, len(self.image_db)))
            img = n[0] / 255
            mask = n[1] / 255
            image_shape = img.shape[0:-1]
            mask_shape = mask.shape
            # for unknown reasons some masks are not of the same size of the images
            if image_shape != mask_shape:
                mask = misc.imresize(mask, image_shape)
                mask = mask / 255  # not clear why we need this but we do
                print('This image does not work {} - Performing resize of the mask'.format(i))
            limitH = (0, image_shape[0] - ih)
            limitW = (0, image_shape[1] - iw)
            for pxi in range(pximage):
                for j in range(10000):
                    h0 = rand.randint(limitH[0], limitH[1])
                    w0 = rand.randint(limitW[0], limitW[1])
                    im11 = img[h0:h0+ih, w0:w0+iw, :]
                    im12 = mask[h0:h0+ih, w0:w0+iw]
                    # One white / one black
                    if np.sum(im12) == p1val:
                        for k in range(10000):
                            h0 = rand.randint(limitH[0], limitH[1])
                            w0 = rand.randint(limitW[0], limitW[1])
                            im21 = img[h0:h0 + ih, w0:w0 + iw, :]
                            im22 = mask[h0:h0 + ih, w0:w0 + iw]
                            if np.sum(im22) == p2val:
                                im1_tensor.append(im11)
                                im2_tensor.append(im21)
                                lb_tensor.append(label)
                                im_names.append(self.textual_db[i])
                                break
                        print('--> break {} at {}/{} --> im1 and im2 size = {}/{}'.format(pxi, j, k,
                                                                                          len(im1_tensor),
                                                                                          len(im2_tensor)))
                        break

        return im1_tensor, im2_tensor, lb_tensor, im_names

    def prepare_data_patches_siamese(self, iw, ih, pximage=10):
        # im1_tensor = list()
        # im2_tensor = list()
        # lb_tensor = list()
        # im_names = list()
        # one white / one black
        print('one white / one black')
        im1_tensor, im2_tensor, lb_tensor, im_names = self._cycle_for_patches(iw, ih,
                                                                              p1val=0,
                                                                              p2val=1,
                                                                              label=1,
                                                                              pximage=pximage)
        # both black
        tmp_im1, tmp_im2, tmp_lb, tmp_names = self._cycle_for_patches(iw, ih,
                                                                      p1val=0,
                                                                      p2val=0,
                                                                      label=0,
                                                                      pximage=pximage)
        im1_tensor += tmp_im1
        im2_tensor += tmp_im2
        lb_tensor += tmp_lb
        im_names += tmp_names
        # both white
        tmp_im1, tmp_im2, tmp_lb, tmp_names = self._cycle_for_patches(iw, ih,
                                                                      p1val=1,
                                                                      p2val=1,
                                                                      label=0,
                                                                      pximage=pximage)
        im1_tensor += tmp_im1
        im2_tensor += tmp_im2
        lb_tensor += tmp_lb
        im_names += tmp_names
        self.textual_db = im_names
        self.x1 = np.asarray(im1_tensor)
        self.x2 = np.asarray(im2_tensor)
        lb_tensor = np.asarray(lb_tensor)
        self.y = np.reshape(lb_tensor, (lb_tensor.shape[0], 1))

    def to_categorical(self):
        tmp = np.zeros_like(self.y)
        np.copyto(tmp, self.y)
        tmp = tmp.astype(dtype=np.int32)
        n = tmp.max() + 1
        y = np.zeros((tmp.shape[0], n), np.float32)
        y[range(0, tmp.shape[0]), tmp] = 1
        return y

    def reset_training(self):
        self.b_pointer = 0
        self.epoch = 1

    def shuffle_db(self):
        n_imgs = self.x1.shape[0]
        ord = [i for i in range(n_imgs)]
        rand.shuffle(ord)
        self.x1 = self.x1[ord, :, :, :]
        self.x2 = self.x2[ord, :, :, :]
        self.y = self.y[ord, :]
        return ord

    def next_batch(self, b_size):
        b_idx = [self.b_pointer, self.b_pointer + b_size]
        # check if the batch is finished or returning 0
        self.b_pointer += b_size
        batch_imgs1 = self.x1[b_idx[0]:b_idx[1], :, :, :]
        batch_imgs2 = self.x2[b_idx[0]:b_idx[1], :, :, :]
        y = self.y[b_idx[0]:b_idx[1], :]
        if len(batch_imgs1) < b_size:
            self.shuffle_db()
            self.b_pointer = 0
            self.epoch += 1
        return batch_imgs1, batch_imgs2, y

    def fetch_random_validation_set_from_training(self, b_size):
        r = rand.randint(0, self.x1.shape[0] - b_size)
        b_idx = [r, r + b_size]
        batch_imgs1 = self.x1[b_idx[0]:b_idx[1], :, :, :]
        batch_imgs2 = self.x2[b_idx[0]:b_idx[1], :, :, :]
        batch_masks = self.y[b_idx[0]:b_idx[1], :]
        return batch_imgs1, batch_imgs2, batch_masks

    def fetch_random_n_samples_x_class(self, n):
        self.shuffle_db()
        n1 = 0
        n2 = 0
        val = list()
        for i in range(self.x1.shape[0]):
            if self.y[i, 1] == 0 and n1 < n:
                val.append(i)
                n1 += 1
            elif self.y[i, 1] == 1 and n2 < n:
                val.append(i)
                n2 += 1
            elif n1 > n and n2 > n:
                break
        x1_val = np.zeros((n*2, self.x1.shape[1], self.x1.shape[2], self.x1.shape[3]))
        x2_val = np.zeros((n*2, self.x2.shape[1], self.x2.shape[2], self.x2.shape[3]))
        y_val = np.zeros((n*2, self.y.shape[1]))
        np.copyto(x1_val, self.x1[val, :, :, :])
        np.copyto(x2_val, self.x2[val, :, :, :])
        np.copyto(y_val, self.y[val, :])
        self.x1 = np.delete(self.x1, val, 0)
        self.x2 = np.delete(self.x2, val, 0)
        self.y = np.delete(self.y, val, 0)
        return x1_val, x2_val, y_val

    def read_from_h5(self, filename):
        with h5py.File(filename, 'r') as fin:
            self.x1 = fin['x1'].value
            self.x2 = fin['x2'].value
            self.y = fin['y'].value


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
        dataset = list()
        for i, name in enumerate(files_tp):
            tmp_file = os.path.join(self.folder_tampered, name)
            mask_file = glob.glob1(self.folder_mask, '{}*'.format(name.split('.')[:-1][0]))
            if len(mask_file) == 0:
                raise Exception('bad news! the mask for {} is missing!'.format(name))
            dataset.append((tmp_file, os.path.join(self.folder_mask, mask_file[0])))
        return dataset

class Main:
    def __init__(self):
        pass

    def main(self):
        t0 = time()
        patch_size = 40
        launch_time = datetime.datetime.now()
        fileout_root = '{0:4}{1:2}{2:2}_{3:2}{4:2}{5:2}_p{6}x{6}.h5'.format(launch_time.year,
                                                                            launch_time.month,
                                                                            launch_time.day,
                                                                            launch_time.hour,
                                                                            launch_time.minute,
                                                                            launch_time.second,
                                                                            patch_size).replace(' ', '0')
        dir_out = 'data'
        fileout = os.path.join(dir_out, fileout_root)


        sets = Settings('config.cfg')
        db_list = sets.check_dataset()
        t1 = time()
        data_reader = DataManager_siamese()
        data_reader.read_data(db_list)
        t2 = time()
        data_reader.prepare_data_patches_siamese(patch_size, patch_size, pximage=10)
        t3 = time()

        with h5py.File(fileout, 'w') as f:
            f.create_dataset('x1', shape=data_reader.x1.shape, dtype=np.float32, data=data_reader.x1)
            f.create_dataset('x2', shape=data_reader.x2.shape, dtype=np.float32, data=data_reader.x2)
            f.create_dataset('y', shape=data_reader.y.shape, dtype=np.float32, data=data_reader.y)
            f.flush()

        print('Time to startup: {}'.format(datetime.timedelta(seconds=t1-t0)))
        print('Time to load dataset: {}'.format(datetime.timedelta(seconds=t2-t1)))
        print('Time to extract patches: {}'.format(datetime.timedelta(seconds=t3-t2)))
        print('Time to save the db: {}'.format(datetime.timedelta(seconds=time()-t3)))

if __name__ == '__main__':
    m = Main()
    m.main()
