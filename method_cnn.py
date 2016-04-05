from keras.optimizers import SGD, Adam
from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import model_from_json
from keras.utils import np_utils
from scipy.ndimage import imread
import numpy as np
from tqdm import tqdm
import time
import os
import myfilelib as MY
import h5py
import matplotlib.pyplot as pl
from random import shuffle
from keras.utils.io_utils import HDF5Matrix
from casiaDB_handler import border_patch_sampling, random_patch_sampling


def VGG_like_convnet(data_shape, opt):
    print('Training VGG net.')
    model = Sequential()
    # input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(data_shape[0], data_shape[1], data_shape[2])))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    # Note: Keras does automatic shape inference.
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(2))
    model.add(Activation('softmax'))

    print ('VGG_like_convnet... nb params: {}'.format(model.count_params()))
    model.compile(loss='categorical_crossentropy', optimizer=opt)
    return model

def VGG_like_convnet_graph(data_shape, opt):
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

    model.add_node(Dense(2), name='dense2', input='relu5')
    model.add_node(Activation('softmax'), name='softmax_out', input='dense2')
    model.add_output(name='class_out', input='softmax_out')
    print ('VGG_like_convnet_graph... nb params: {}'.format(model.count_params()))
    model.compile(loss={'class_out':'categorical_crossentropy'}, optimizer=opt)
    return model


def AlexNet_like_convnet(data_shape, opt):
    print('Training AlexNet net.')
    model = Sequential()
    model.add(Convolution2D(96, 10, 10, border_mode='valid', input_shape=(data_shape[0], data_shape[1], data_shape[2])))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(256, 5, 5, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(BatchNormalization(epsilon=1e-06, mode=0))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(300, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(BatchNormalization(epsilon=1e-06, mode=0))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, init='normal'))
    model.add(Activation('relu'))
    model.add(BatchNormalization(epsilon=1e-06, mode=0))
    model.add(Dense(256, init='normal'))
    model.add(Activation('relu'))
    model.add(BatchNormalization(epsilon=1e-06, mode=0))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    print ('AlexNet_like_convnet... nb params: {}'.format(model.count_params()))
    model.compile(loss='categorical_crossentropy', optimizer=opt)
    return model


def read_model_from_disk(weights_path, model_path):
    assert os.path.exists(model_path), 'Model json not found (see "model_path" variable in script).'
    model = model_from_json(open(model_path).read())
    assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
    with h5py.File(weights_path) as f:
        for k in range(f.attrs['nb_layers']):
            if k >= len(model.layers):
                # we don't look at the last (fully-connected) layers in the savefile
                break
            g = f['layer_{}'.format(k)]
            weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
            model.layers[k].set_weights(weights)
    print('Model loaded.')
    return model


def train_cnn(training_h5, test_h5, settings):
    with h5py.File(training_h5, 'r') as f:
        training_size = f.values()[0].shape
        print('Training size: {}'.format(training_size))
    with h5py.File(test_h5, 'r') as f:
        test_size = f.values()[0].shape

    # Training model
    modelfileweights = os.path.join(settings.working_folder, 'model{2}_weights_ep{0:02d}_bs{1:02d}.h5'.format(settings.nb_epochs, settings.batch_size, settings.method))
    modelfilename = os.path.join(settings.working_folder, 'model{2}_ep{0:02d}_bs{1:02d}.json'.format(settings.nb_epochs, settings.batch_size, settings.method))
    if not(os.path.exists(modelfileweights) & os.path.exists(modelfilename)):
        # sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        adam = Adam()
        if settings.method == 'CNN_VGG':
            model = VGG_like_convnet((training_size[1], training_size[2], training_size[3]), adam)
        elif settings.method == 'CNN_ALEX':
            model = AlexNet_like_convnet((training_size[1], training_size[2], training_size[3]), adam)
        nb_params = model.count_params()

        # Get data out of the H5 dataset =
        train_x = HDF5Matrix(training_h5, 'data', 0, training_size[0])
        train_y = HDF5Matrix(training_h5, 'label', 0, training_size[0])
        test_x = HDF5Matrix(test_h5, 'data', 0, training_size[0])
        test_y = HDF5Matrix(test_h5, 'label', 0, training_size[0])

        # Train model
        t3 = time.time()

        model.fit(train_x, train_y, batch_size=settings.batch_size, nb_epoch=settings.nb_epochs, validation_data=(test_x, test_y), shuffle='batch', show_accuracy=True)
        # model.fit({'data_in':train_x, 'class_out':train_y}, batch_size=settings.batch_size, nb_epoch=settings.nb_epochs, validation_data={'data_in':test_x, 'class_out':test_y})

        # Save the model
        t4 = time.time()
        json_string = model.to_json()
        open(modelfilename, 'w').write(json_string)
        model.save_weights(modelfileweights, overwrite=True)
    else:
        print('Read model from file.')
        model = read_model_from_disk(modelfileweights, modelfilename)
    return model


def test_cnn(test_images, model, batch_size=256, useBorders = 0):
    ###### Test model #####
    tinit = time.time()
    results = np.zeros((len(test_images), 1))
    for i in range(len(test_images)):
        img = imread(test_images[i].image_path, mode='RGB')
        # img = cv2.imread(test_images[i].image_path, flags=cv2.IMREAD_COLOR)
        if useBorders:
            img_b = imread(test_images[i].border_image, flatten=True)
            test_x = border_patch_sampling(img, patch_size=40, stride=20, b_image=img_b, b_thr=1)
            if len(test_x) == 0:
                test_x = random_patch_sampling(img, patch_size=40, stride=20, howmany=11)
        else:
            test_x = border_patch_sampling(img, patch_size=40, stride=20)

        # Normalization
        test_x = test_x / 255
        # prediction = model.predict_classes({'data_in':test_x}, batch_size=settings.batch_size, verbose=True)
        # nb_0 = len(prediction['class_out'].argwhere(0))
        # nb_1 = len(prediction['class_out'].argwhere(1))
        prediction = model.predict_classes(test_x, batch_size=batch_size, verbose=True)
        nb_0 = len(np.argwhere(prediction == 0))
        nb_1 = len(np.argwhere(prediction == 1))
        if nb_0 > nb_1:
            pred = 0
        else:
            pred = 1
        results[i, 0] = pred

    tend = time.time()
    # print('Time get training samples: {}'.format(MY.hms_string(t1 - t0)))
    # print('Time get test samples: {}'.format(MY.hms_string(t2 - t1)))
    # print('Time to generate the model: {}'.format(MY.hms_string(t3 - t2)))
    # print('Time for training the model: {}'.format(MY.hms_string(t4 - t3)))
    # print('Time for saving the model: {}'.format(MY.hms_string(t5 - t4)))
    print('Time for testing model: {}'.format(MY.hms_string(tend - tinit)))
    return results
