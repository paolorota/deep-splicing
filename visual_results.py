import ConfigParser
import os, sys, glob
import random
import numpy as np
import time
from method_cnn import train_cnn, test_cnn
import pandas as pd
import casiaDB_handler as casia
import myfilelib as MY
from auccreator import getAUC
from deep_tester import Settings, readtestfile
from method_cnn import read_model_from_disk
from scipy.ndimage import imread
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def test_myimage(img, model, batch_size=1, img_b=None, mask=None, patch_size=40, stride=20, beta=1):
    (rows, cols, channels) = img.shape
    bias_rows = int(round((rows % patch_size) / 2))
    bias_cols = int(round((cols % patch_size) / 2))
    auth_patch_list = []
    tamp_patch_list = []
    out_image = np.copy(img)
    out_true = np.copy(img)
    out_true_mask = np.zeros(mask.shape)
    out_pred_mask = np.zeros(mask.shape)
    # put m_image from 255/0 to 1/0
    if not(mask == None):
        nonzeros = np.argwhere(mask > 0)
        mask[nonzeros[:, 0], nonzeros[:, 1]] = 1
    for r in range(bias_rows, rows - bias_rows - patch_size, stride):
        for c in range(bias_cols, cols - bias_cols - patch_size, stride):
            isTampered = False
            m_patch = mask[r:r + patch_size, c:c + patch_size]
            nb_whites = m_patch.sum()
            if nb_whites > float(patch_size) * beta and nb_whites < (patch_size ** 2) - float(patch_size) * beta:
                # is tampered
                isTampered = True
            elif nb_whites >= (patch_size ** 2) - float(patch_size) * beta:
                # not considering the patch for training if it is in the tampered area
                continue
            # now I know the ground truth
            patch = img[r:r + patch_size, c:c + patch_size, :]
            patch = patch.reshape(1, patch_size, patch_size, 3)
            patch = np.rollaxis(patch, 3, 1)
            pred = model.predict_classes(patch, batch_size=batch_size, verbose=0)
            if pred == 1:
                # increase the red channel of 100
                out_image[r:r + patch_size, c:c + patch_size, 0] += 100
                out_pred_mask[r:r + patch_size, c:c + patch_size] = 1

            if isTampered:
                out_true[r:r + patch_size, c:c + patch_size, 0] += 100
                out_true_mask[r:r + patch_size, c:c + patch_size] = 1

    #PRINT THEM ALL
    fig = plt.figure()
    a=fig.add_subplot(2, 3,1)
    imgplot = plt.imshow(out_true)
    a.set_title('True')
    plt.axis('off')
    a = fig.add_subplot(2, 3, 2)
    imgplot = plt.imshow(out_image)
    a.set_title('Predicted')
    plt.axis('off')
    a = fig.add_subplot(2, 3, 3)
    imgplot = plt.imshow(mask, cmap='Greys')
    a.set_title('Mask')
    plt.axis('off')
    a = fig.add_subplot(2, 3, 4)
    imgplot = plt.imshow(out_pred_mask, cmap='Greys')
    a.set_title('Pred Mask')
    plt.axis('off')
    a = fig.add_subplot(2, 3, 5)
    imgplot = plt.imshow(out_true_mask, cmap='Greys')
    a.set_title('Expected Mask')
    plt.axis('off')
    plt.show()





def main():
    # Params
    if (len(sys.argv) == 2):
        settingsFileName = sys.argv[1]
    else:
        settingsFileName = 'config.ini'
    settings = Settings(settingsFileName)
    results_dir = os.path.join(settings.working_folder, 'results')
    isDebug = 1

    try:
        os.stat(results_dir)
    except:
        print('Creating results folder: {}'.format(results_dir))
        os.mkdir(results_dir)

    # search for training and test files
    test_filelist = glob.glob1(settings.working_folder, 'test*')

    nb_tests = len(test_filelist)
    if isDebug == 1:
        nb_tests = 1

    modelfileweights = os.path.join(settings.working_folder,
                                    'model{2}_b{3}_weights_ep{0:02d}_bs{1:02d}.h5'.format(settings.nb_epochs,
                                                                                          settings.batch_size,
                                                                                          settings.method,
                                                                                          settings.use_borders))
    modelfilename = os.path.join(settings.working_folder,
                                 'model{2}_b{3}_ep{0:02d}_bs{1:02d}.json'.format(settings.nb_epochs,
                                                                                 settings.batch_size,
                                                                                 settings.method,
                                                                                 settings.use_borders))
    for t in range(nb_tests):

        test_images = readtestfile(os.path.join(settings.working_folder, test_filelist[t]), settings)
        model = read_model_from_disk(modelfileweights, modelfilename)
        for i in test_images:
            img = imread(i.image_path, mode='RGB')
            mask = imread(i.mask_image, flatten=True)
            test_myimage(img, model, mask=mask)

########### END ################

if __name__ == '__main__':
    main()