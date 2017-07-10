from os.path import join
from scipy import misc
import cv2
import numpy as np
import h5py
from matplotlib import pyplot as plt

w_dir = '/home/prota/Datasets'
size_imgs = 128
training_file_list = join(w_dir, 'training.txt')
test_file_list = join(w_dir, 'test.txt')

# prepare training set
with open(training_file_list, 'r') as fin:
    tmp_file = fin.read()

tmp_file = tmp_file.split('\n')
tmp_file = tmp_file[:-1]
training_set_img = np.zeros((len(tmp_file), size_imgs, size_imgs, 3), dtype=np.float32)
training_set_cnt = np.zeros((len(tmp_file), size_imgs, size_imgs, 1), dtype=np.float32)
for i, n in enumerate(tmp_file):
    try:
        if i % 1000 == 0:
            print('training {}/{}'.format(i, len(tmp_file)))
        img = misc.imread(n)
        img = misc.imresize(img, (size_imgs, size_imgs, 3))
        sb = cv2.Canny(img, 100, 200)
        sb = misc.imresize(sb, (size_imgs, size_imgs))
        training_set_img[i, :, :, :] = img
        training_set_cnt[i, :, :, 0] = sb
        # plt.subplot(121), plt.imshow(img)
        # plt.subplot(122), plt.imshow(sb, cmap='gray')
        # plt.show()
    except ValueError:
        print('FOUND WRONG FILE: {}'.format(n))
        continue
    except AttributeError:
        print('Problem with the file {}'.format(n))

# prepare test set
with open(test_file_list, 'r') as fin:
    tmp_file = fin.read()

tmp_file = tmp_file.split('\n')
tmp_file = tmp_file[:-1]
test_set_img = np.zeros((len(tmp_file), size_imgs, size_imgs, 3), dtype=np.float32)
test_set_cnt = np.zeros((len(tmp_file), size_imgs, size_imgs, 1), dtype=np.float32)
for i, n in enumerate(tmp_file):
    try:
        if i % 1000 == 0:
            print('test {}/{}'.format(i, len(tmp_file)))
        img = misc.imread(n)
        img = misc.imresize(img, (size_imgs, size_imgs, 3))
        sb = cv2.Canny(img, 100, 200)
        sb = misc.imresize(sb, (size_imgs, size_imgs))
        test_set_img[i, :, :, :] = img
        test_set_cnt[i, :, :, 0] = sb
    except ValueError:
        print('FOUND WRONG FILE: {}'.format(n))
        continue
    except AttributeError:
        print('Problem with the file {}'.format(n))

with h5py.File(join(w_dir, 'utzap.h5'), 'w') as f:
    f.create_dataset('train_img', shape=training_set_img.shape, dtype=np.float32, data=training_set_img)
    f.create_dataset('train_canny', shape=training_set_cnt.shape, dtype=np.float32, data=training_set_cnt)
    f.create_dataset('test_img', shape=test_set_img.shape, dtype=np.float32, data=test_set_img)
    f.create_dataset('test_canny', shape=test_set_cnt.shape, dtype=np.float32, data=test_set_cnt)
    f.flush()