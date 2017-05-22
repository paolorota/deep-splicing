import os
import h5py
from scipy import misc
import glob
import numpy as np
from matplotlib import pyplot as plt

join = os.path.join

data_dir = '/home/prota/Datasets/celebA/'
fileout = './data/celebA.h5'
file_list = glob.glob1(data_dir, '*33.jpg')

imglist = np.zeros((len(file_list), 128, 128, 3), np.float32)

for i, n in enumerate(file_list):
    print('Processing file {}/{}: {}'.format(i+1, len(file_list), n))
    img = misc.imread(join(data_dir, n))
    img_shape = (img.shape[0]//2 - 64, img.shape[1]//2 - 64)
    img = img[img_shape[0]:img_shape[0] + 128, img_shape[1]:img_shape[1] + 128, :]
    imglist[i] = img / 255
    # plt.imshow(imglist[i])
    # plt.show()

with h5py.File(fileout, 'w') as f:
    f.create_dataset('x', shape=imglist.shape, dtype=np.float32, data=imglist)
    f.create_dataset('y', shape=imglist.shape, dtype=np.float32, data=imglist)
    f.flush()