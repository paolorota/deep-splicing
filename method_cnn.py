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


def VGG_like_convnet(data_shape, opt):
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
    parray = np.zeros((nb_patches, 3, patch_size, patch_size), dtype=np.float32)
    for i in range(nb_patches):
        img = patch_list[i]
        new_img = np.rollaxis(img, 2, 0)
        parray[i, :, :, :] = new_img
    return parray

def get_patch_array(myimages, description, p_size, p_stride):
    x_tmp = []
    y_tmp = []
    nb_over_patches = 0
    for i in tqdm(range(len(myimages)), desc=description):
        # Load an color image in BGR
        img = imread(myimages[i].image_path, mode='RGB')
        # img = cv2.imread(myimages[i].image_path, flags=cv2.IMREAD_COLOR)
        tmp_plist = exhaustive_patch_sampling(img, patch_size=p_size, stride=p_stride)
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
    #print('DEBUG: pshape: {} && x_arr: {}'.format(p_shape, x_arr.shape))
    c = 0
    for i in tqdm(range(len(myimages)), desc='Conversion to array'):
        single_image_patches = x_tmp[i]
        single_image_targets = y_tmp[i]
        pack_size = len(single_image_patches)
        x_arr[c:c+pack_size, :, :, :] = single_image_patches
        y_arr[c:c+pack_size, :] = single_image_targets
        c += pack_size
    #print('DEBUG: pack_size: {} (must be: {})'.format(c, nb_over_patches))
    return x_arr, y_arr


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


def run_cnn(training_images, test_images, settings, test_number):
    nb_training = len(training_images)
    nb_test = len(test_images)
    doLoading = False # used just for printing times in the end

    # Training model
    modelfileweights = os.path.join(settings.working_folder, 'modelNN_weights_ep{0:02d}_bs{1:02d}.h5'.format(settings.nb_epochs, settings.batch_size))
    modelfilename = os.path.join(settings.working_folder, 'modelNN_ep{0:02d}_bs{1:02d}.json'.format(settings.nb_epochs, settings.batch_size))
    if not(os.path.exists(modelfileweights) & os.path.exists(modelfilename)):
        # extract patches for training
        t0 = time.time()
        tmp_filename = 'tmp_training_ts{}.h5'.format(test_number)
        tmp_filename = os.path.join(settings.working_folder, tmp_filename)
        if os.path.exists(tmp_filename):
            print('{} found! No need to fetch data again.'.format(tmp_filename))
            with h5py.File(tmp_filename, 'r') as f:
                train_x = f['data'].value
                train_y = f['label'].value
        else:
            print('{} not found! Need to fetch data.'.format(tmp_filename))
            train_x, train_y = get_patch_array(training_images, 'Creation of training set', settings.patch_size,
                                               settings.patch_stride)
            print('Saving Training set: {}'.format(tmp_filename))
            with h5py.File(tmp_filename, 'w') as f:
                f.create_dataset('data', data=train_x)
                f.create_dataset('label', data=train_y)
                f.flush()
        train_y = np_utils.to_categorical(train_y, 2)

        # extract patches for test
        t1 = time.time()
        tmp_filename = 'tmp_test_ts{}.h5'.format(test_number)
        tmp_filename = os.path.join(settings.working_folder, tmp_filename)
        if os.path.exists(tmp_filename):
            print('{} found! No need to fetch data again.'.format(tmp_filename))
            with h5py.File(tmp_filename, 'r') as f:
                test_x = f['data'].value
                test_y = f['label'].value
        else:
            print('{} not found! Need to fetch data.'.format(tmp_filename))
            test_x, test_y = get_patch_array(test_images, 'Creation of test set', settings.patch_size,
                                             settings.patch_stride)  # Non necessario se non per validation
            print('Saving Test set: {}'.format(tmp_filename))
            with h5py.File(tmp_filename, 'w') as f:
                f.create_dataset('data', data=test_x)
                f.create_dataset('label', data=test_y)
                f.flush()
        test_y = np_utils.to_categorical(test_y, 2)

        # Normalization
        train_x = train_x / 255
        test_x = test_x / 255

        # Create Model
        t2 = time.time()
        # sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        adam = Adam()
        model = VGG_like_convnet((train_x.shape[1], train_x.shape[2], train_x.shape[3]), adam)
        nb_params = model.count_params()

        # Train model
        t3 = time.time()

        model.fit(train_x, train_y, batch_size=settings.batch_size, nb_epoch=settings.nb_epochs, validation_data=(test_x, test_y))
        # model.fit({'data_in':train_x, 'class_out':train_y}, batch_size=settings.batch_size, nb_epoch=settings.nb_epochs, validation_data={'data_in':test_x, 'class_out':test_y})

        # Save the model
        t4 = time.time()
        json_string = model.to_json()
        open(modelfilename, 'w').write(json_string)
        model.save_weights(modelfileweights, overwrite=True)
    else:
        print('Read model from file.')
        model = read_model_from_disk(modelfileweights, modelfilename)

    ###### Test model #####
    t5 = time.time()
    results = np.zeros((nb_test, 1))
    for i in range(nb_test):
        img = imread(test_images[i].image_path, mode='RGB')
        # img = cv2.imread(test_images[i].image_path, flags=cv2.IMREAD_COLOR)
        test_x = exhaustive_patch_sampling(img, patch_size=40, stride=20)
        # Normalization
        test_x = test_x / 255
        # prediction = model.predict_classes({'data_in':test_x}, batch_size=settings.batch_size, verbose=True)
        # nb_0 = len(prediction['class_out'].argwhere(0))
        # nb_1 = len(prediction['class_out'].argwhere(1))
        prediction = model.predict_classes(test_x, batch_size=settings.batch_size, verbose=True)
        nb_0 = len(np.argwhere(prediction == 0))
        nb_1 = len(np.argwhere(prediction == 1))
        if nb_0 > nb_1:
            pred = 0
        else:
            pred = 1
        results[i, 0] = pred

    t6 = time.time()
    # print('Time get training samples: {}'.format(MY.hms_string(t1 - t0)))
    # print('Time get test samples: {}'.format(MY.hms_string(t2 - t1)))
    # print('Time to generate the model: {}'.format(MY.hms_string(t3 - t2)))
    # print('Time for training the model: {}'.format(MY.hms_string(t4 - t3)))
    # print('Time for saving the model: {}'.format(MY.hms_string(t5 - t4)))
    # print('Time for testing model: {}'.format(MY.hms_string(t6 - t5)))
    return results
