import os
import numpy as np
import h5py
from tqdm import tqdm
from scipy.ndimage import imread
from random import shuffle
from keras.utils import np_utils


def border_patch_sampling(image, patch_size=40, stride=20, b_image=None, b_thr=1):
    (rows, cols, channels) = image.shape
    bias_rows = int(round((rows % patch_size) / 2))
    bias_cols = int(round((cols % patch_size) / 2))
    patch_list = []
    for r in range(bias_rows, rows - bias_rows - patch_size, stride):
        for c in range(bias_cols, cols - bias_cols - patch_size, stride):
            # check if the border image has a border inside
            if not(b_image == None):
                b_patch = b_image[r:r + patch_size, c:c + patch_size]
                b_patch[b_patch <= 128] = 0
                if len(b_patch[b_patch == 0]) > float(b_thr) * float(patch_size):
                    patch = image[r:r + patch_size, c:c + patch_size, :]
                    patch_list.append(patch)
            # don't care about borders in this case, get all the patches
            else:
                patch = image[r:r + patch_size, c:c + patch_size, :]
                patch_list.append(patch)
    # now list containing image as (rows, cols, channels)
    # need to be exported as a ndarray (n_samples, n_channels, n_rows, n_cols)
    nb_patches = len(patch_list)
    parray = np.zeros((nb_patches, 3, patch_size, patch_size), dtype=np.float32)
    for i in range(nb_patches):
        img = patch_list[i]
        new_img = np.rollaxis(img, 2, 0)
        parray[i, :, :, :] = new_img
    return parray


def random_patch_sampling(image, patch_size=40, stride=20, howmany=10):
    (rows, cols, channels) = image.shape
    bias_rows = int(round((rows % patch_size) / 2))
    bias_cols = int(round((cols % patch_size) / 2))
    patch_list = []
    for r in range(bias_rows, rows - bias_rows - patch_size, stride):
        for c in range(bias_cols, cols - bias_cols - patch_size, stride):
            patch = image[r:r + patch_size, c:c + patch_size, :]
            patch_list.append(patch)
    # now list containing image as (rows, cols, channels)
    # need to be exported as a ndarray (n_samples, n_channels, n_rows, n_cols)
    nb_patches = len(patch_list)
    shuffle(patch_list)
    patch_list = patch_list[0:howmany]
    nb_patches = len(patch_list)
    parray = np.zeros((nb_patches, 3, patch_size, patch_size), dtype=np.float32)
    for i in range(nb_patches):
        img = patch_list[i]
        new_img = np.rollaxis(img, 2, 0)
        parray[i, :, :, :] = new_img
    return parray


def get_patch_array(myimages, description, p_size, p_stride, doBorderSearch=1):
    x_tmp = []
    y_tmp = []
    nb_over_patches = 0
    for i in tqdm(range(len(myimages)), desc=description):
        # Load an color image in BGR
        img = imread(myimages[i].image_path, mode='RGB')
        if doBorderSearch:
            img_b = imread(myimages[i].border_image, flatten=True)
            tmp_plist = border_patch_sampling(img, patch_size=p_size, stride=p_stride, b_image=img_b)
        else:
            tmp_plist = border_patch_sampling(img, patch_size=p_size, stride=p_stride)
        nb_samples = int(tmp_plist.shape[0])
        if myimages[i].label == 0:
            tmp_labels = np.zeros((nb_samples, 1), dtype=np.float32)
        elif myimages[i].label == 1:
            tmp_labels = np.ones((nb_samples, 1), dtype=np.float32)
        else:
            raise ("A label different fro 0 and 1 is not possible.")
        # Creation of the training set
        x_tmp.append(tmp_plist)
        y_tmp.append(tmp_labels)
        nb_over_patches += len(tmp_labels)
    # generate the final array
    p_shape = x_tmp[0].shape
    x_arr = np.zeros((nb_over_patches, p_shape[1], p_shape[2], p_shape[3]), dtype=np.float32)
    y_arr = np.zeros((nb_over_patches, 1), dtype=np.float32)
    c = 0
    for i in tqdm(range(len(myimages)), desc='Conversion to array'):
        single_image_patches = x_tmp[i]
        single_image_targets = y_tmp[i]
        pack_size = len(single_image_patches)
        x_arr[c:c+pack_size, :, :, :] = single_image_patches
        y_arr[c:c+pack_size, :] = single_image_targets
        c += pack_size
    return x_arr, y_arr

## Creates the h5 dataset for training and test (test might be used for validation only since it is patch based)
def create_database(training_images, test_images, prename='tmp', patch_size=40, patch_stride=20, working_dir='.', useBorders=0):
    # Creating model
    tmp_filename_train = '{}_training_p{}_s{}.h5'.format(prename, patch_size, patch_stride)
    tmp_filename_train = os.path.join(working_dir, tmp_filename_train)
    if not(os.path.exists(tmp_filename_train)):
        print('{} not found! Need to fetch data.'.format(tmp_filename_train))
        train_x, train_y = get_patch_array(training_images, 'Creation of training set', patch_size,
                                           patch_stride, doBorderSearch=useBorders)
        train_y = np_utils.to_categorical(train_y, 2)
        print('Saving Training set: {}'.format(tmp_filename_train))
        with h5py.File(tmp_filename_train, 'w') as f:
            f.create_dataset('data', data=train_x, dtype='float32')
            f.create_dataset('label', data=train_y, dtype='float32')
            f.flush()
    else:
        print('Loading DB training: {}'.format(tmp_filename_train))

    # extract patches for test
    tmp_filename_test = '{}_test_p{}_s{}.h5'.format(prename, patch_size, patch_stride)
    tmp_filename_test = os.path.join(working_dir, tmp_filename_test)
    if not(os.path.exists(tmp_filename_test)):
        print('{} not found! Need to fetch data.'.format(tmp_filename_test))
        test_x, test_y = get_patch_array(test_images, 'Creation of test set', patch_size,
                                         patch_stride,
                                         doBorderSearch=useBorders)  # Non necessario se non per validation
        test_y = np_utils.to_categorical(test_y, 2)
        print('Saving Test set: {}'.format(tmp_filename_test))
        with h5py.File(tmp_filename_test, 'w') as f:
            f.create_dataset('data', data=test_x, dtype='float32')
            f.create_dataset('label', data=test_y, dtype='float32')
            f.flush()
    else:
        print('Loading DB test: {}'.format(tmp_filename_test))
    return tmp_filename_train, tmp_filename_test