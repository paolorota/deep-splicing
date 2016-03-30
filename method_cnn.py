from keras.optimizers import SGD, adam
from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten, Merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import cv2
import numpy as np
from tqdm import tqdm


def VGG_regression_net(data_shape):
    # create a network
    model = Sequential()
    # input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(data_shape[0], data_shape[1], data_shape[2])))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization(epsilon=5e-5, momentum=0.75))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization(epsilon=5e-5, momentum=0.75))
    model.add(Dropout(0.25))

    model.add(Flatten())
    # Note: Keras does automatic shape inference.
    model.add(Dense(256))
    model.add(Activation('relu'))

    model.add(Dense(1))
    model.add(Activation('relu'))
    print ('VGG_regression_net... nb params: {}'.format(model.count_params()))
    return model

def VGG_regression_net_graph(data_shape):
    # create a network
    model = Graph()
    model.add_input(name='data_in', input_shape=(data_shape[0], data_shape[1], data_shape[2]))
    model.add_node(Convolution2D(32, 3, 3, border_mode='valid'), name='conv1', input='data_in')
    model.add_node(Activation('relu'), name='relu1', input='conv1')
    model.add_node(Convolution2D(32, 3, 3), name='conv2', input='relu1')
    model.add_node(Activation('relu'), name='relu2', input='conv2')
    model.add_node(MaxPooling2D(pool_size=(2, 2)), name='pool1', input='relu2')
    # model.add(Dropout(0.25))

    model.add_node(Convolution2D(64, 3, 3, border_mode='valid'), name='conv3', input='pool1')
    model.add_node(Activation('relu'), name='relu3', input='conv3')
    model.add_node(Convolution2D(64, 3, 3), name='conv4', input='relu3')
    model.add_node(Activation('relu'), name='relu4', input='conv4')
    model.add_node(MaxPooling2D(pool_size=(2, 2)), name='pool2', input='relu4')
    # model.add(Dropout(0.25))

    model.add_node(Flatten(), name='flatten', input='pool2')
    model.add_node(Dense(256), name='dense1', input='flatten')
    model.add_node(Activation('relu'), name='relu5', input='dense1')

    model.add_node(Dense(1), name='dense2', input='relu5')
    model.add_node(Activation('relu'), name='reluout', input='dense2')
    model.add_output(name='output', input='reluout')
    print ('VGG_regression_net_graph... nb params: {}'.format(model.count_params()))
    return model


def exhaustive_patch_sampling(image, patch_size = 40, stride = 20):
    (rows, cols, channels) = image.shape
    bias_rows = int(round((rows % patch_size)/2))
    bias_cols = int(round((cols % patch_size)/2))
    patch_list = []
    for r in range(bias_rows, rows - bias_rows - patch_size, stride):
        for c in range(bias_cols, cols - bias_cols - patch_size, stride):
            patch = image[r:r+patch_size, c:c+patch_size, :]
            patch_list.append(patch)
    # now list containing image as (rows, cols, channels)
    # need to be exported as a ndarray (n_samples, n_channels, n_rows, n_cols)
    nb_patches = len(patch_list)
    parray = np.zeros((nb_patches, 3, patch_size, patch_size))
    for i in range(nb_patches):
        img = patch_list[i]
        new_img = np.rollaxis(img, 2, 0)
        parray[i, :, :, :] = new_img
    return parray


def run_cnn(training_images, test_images, settings):
    nb_training = len(training_images)
    nb_test = len(test_images)

    # extract patches for training
    for i in tqdm(range(nb_training), desc='Creation of training set'):
        # Load an color image in BGR
        img = cv2.imread(training_images[i].image_path,flags=cv2.IMREAD_COLOR)
        tmp_plist = exhaustive_patch_sampling(img, patch_size=40, stride=20)
        nb_samples = int(tmp_plist.shape[0])
        if training_images[i].label == 0:
            tmp_labels = np.zeros((nb_samples, 1))
        elif training_images[i].label == 1:
            tmp_labels = np.ones((nb_samples, 1))
        else:
            raise("A label different fro 0 and 1 is not possible.")
        # Creation of the training set
        if i == 0:
            train_x = tmp_plist
            train_y = tmp_labels
        else:
            train_x = np.concatenate((train_x, tmp_plist))
            train_y = np.concatenate((train_y, tmp_labels))

    # extract patches for test
    for i in tqdm(range(nb_test), desc='Creation of test set'):
        # Load an color image in BGR
        img = cv2.imread(test_images[i].image_path, flags=cv2.IMREAD_COLOR)
        tmp_plist = exhaustive_patch_sampling(img, patch_size=40, stride=20)
        nb_samples = int(tmp_plist.shape[0])
        if test_images[i].label == 0:
            tmp_labels = np.zeros((nb_samples, 1))
        elif test_images[i].label == 1:
            tmp_labels = np.ones((nb_samples, 1))
        else:
            raise ("A label different fro 0 and 1 is not possible.")
        # Creation of the training set
        if i == 0:
            test_x = tmp_plist
            test_y = tmp_labels
        else:
            test_x = np.concatenate((test_x, tmp_plist))
            test_y = np.concatenate((test_y, tmp_labels))