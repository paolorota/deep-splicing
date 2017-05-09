import tools as T
import glob
import os
import sys
import numpy as np
from scipy import misc

def check_dataset(ft, fm):
    '''
    create the list of file
    :return: dataset --> list of tuple(file_path, mask_path)
    '''
    files_tp = glob.glob1(ft, '*')
    files_mask = glob.glob1(fm, '*')
    if len(files_mask) != len(files_tp):
        print('n_tp: {}\nn_mask: {}\n'.format(len(files_tp), len(files_mask)))
        raise Exception('Files mismatch between tp and mask')
    dataset = list()
    for i, name in enumerate(files_tp):
        tmp_file = os.path.join(ft, name)
        mask_file = glob.glob1(fm, '{}*'.format(name.split('.')[:-1][0]))
        if len(mask_file) == 0:
            raise Exception('bad news! the mask for {} is missing!'.format(name))
        dataset.append((tmp_file, os.path.join(fm, mask_file[0])))
    return dataset

path_img = '/home/prota/Datasets/test_db/Tp'
path_mas = '/home/prota/Datasets/test_db/Tp_mask'
out_dir = '/home/prota/tmp'

data_tuple = check_dataset(path_img, path_mas)
dr = T.DataHandler_IFS_TC()
dr.read_data(data_tuple)
dr.prepare_data_resize(iw=300, ih=300)


for n in range(30):
    print('iter {}'.format(n))
    dirname = os.path.join(out_dir, 'iter_{}'.format(n))
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    a, b = dr.next_batch(3)
    # c = list()
    for i in range(len(a)):
        m = np.clip(a[i] + b[i], 0, 255)
        new_size = list(a[i].shape)
        new_size[0] += new_size[0]
        img = np.zeros(new_size, dtype=np.uint8)
        img[:m.shape[0], :, :] = a[i]
        img[m.shape[0]:, :, :] = m
        # c.append(m)
        filename = 'img_b{0:2}_i{1:2}.png'.format(n, i)
        filename = os.path.join(dirname, filename)
        misc.imsave(filename, img)

print('fine')
