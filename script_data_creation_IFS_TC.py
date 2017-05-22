import os
import configparser
import glob
from scipy import misc
import numpy as np
import cv2
import random as rand
from time import time
import datetime
import tools as T

import h5py


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


class DataHandler_IFS_TC(T.DataHandler):
    def __init__(self):
        T.DataHandler.__init__(self)
        self.textual_db = None
        self.image_db = None

    def read_data(self, db):
        self.textual_db = db
        self.image_db = []
        for i, n in enumerate(db):
            im1 = misc.imread(n[0], mode='RGB')
            im2 = misc.imread(n[1], flatten=True)
            self.image_db.append((im1, im2))

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
                mask = mask / 255  # not clear why we need this but we do
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
                im_tensor[myiter] = im1
                mk_tensor[myiter, :, :, 0] = im2
                im_names.append(self.textual_db[i])
                myiter += 1
                if myiter % pximage == 0:
                    break
        self.textual_db = im_names
        self.x = im_tensor
        self.y = mk_tensor

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
        self.x = im_tensor
        self.y = mk_tensor

    def shuffle_db(self):
        ord = T.DataHandler.shuffle_db(self)
        tmp = [self.textual_db[i] for i in ord]
        self.textual_db = tmp



t0 = time()
patch_size = 128
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
data_reader = DataHandler_IFS_TC()
data_reader.read_data(db_list)
t2 = time()
data_reader.prepare_data_patches(patch_size, patch_size, pximage=10)
t3 = time()

with h5py.File(fileout, 'w') as f:
    f.create_dataset('x', shape=data_reader.x.shape, dtype=np.float32, data=data_reader.x)
    f.create_dataset('y', shape=data_reader.y.shape, dtype=np.float32, data=data_reader.y)
    f.flush()

print('Time to startup: {}'.format(datetime.timedelta(seconds=t1-t0)))
print('Time to load dataset: {}'.format(datetime.timedelta(seconds=t2-t1)))
print('Time to extract patches: {}'.format(datetime.timedelta(seconds=t3-t2)))
print('Time to save the db: {}'.format(datetime.timedelta(seconds=time()-t3)))


