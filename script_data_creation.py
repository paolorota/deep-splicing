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
data_reader = T.DataHandler_IFS_TC()
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


