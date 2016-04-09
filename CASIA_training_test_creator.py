import time
import os, sys
import glob
import ConfigParser
import myfilelib as my
from tqdm import tqdm
import numpy as np
from random import shuffle


class Settings:
    def __init__(self, filename):
        print('Reading params form ' + filename)
        Config = ConfigParser.ConfigParser()
        Config.read(filename)
        self.tampered_folder = Config.get('Dataset', 'DB_folder_tp')
        self.authentic_folder = Config.get('Dataset', 'DB_folder_au')
        self.borders_tampered_folder = Config.get('Dataset', 'DB_folder_tp_borders')
        self.borders_authentic_folder = Config.get('Dataset', 'DB_folder_au_borders')
        self.mask_tampered_folder = Config.get('Dataset', 'DB_folder_tp_mask')
        self.working_folder = Config.get('Dataset', 'Working_folder')
        self.pct_test = float(Config.get('Dataset', 'Percent_test')) / 100
        self.kfold = int(Config.get('Dataset', 'K_fold'))

def getBorderFile(filename, dirborder):
    mydir, fname, exp = my.fileparts(filename)
    bordername = '{}_*'.format(fname)
    bordername = glob.glob(os.path.join(dirborder, bordername))[0]
    mydir1, fname1, exp1 = my.fileparts(bordername)
    spl = fname1.split('_t')
    thr = float(spl[-1])
    fname1 = '{}.{}'.format(fname1, exp1)
    return fname1, thr


def getMaskedFile(filename, dirmask):
    mydir, fname, exp = my.fileparts(filename)
    maskname = '{}_*'.format(fname)
    maskname = glob.glob(os.path.join(dirmask, maskname))[0]
    mydir1, fname1, exp1 = my.fileparts(maskname)
    spl = fname1.split('_b')
    isBlurred = bool(spl[-1])
    fname1 = '{}.{}'.format(fname1, exp1)
    return fname1, isBlurred


def printFileList(fullname, mylist):
    with open(fullname, mode='w') as fin:
        for i in range(len(mylist)):
            fin.write(mylist[i] + '\n')


def main():
    # Params
    if (len(sys.argv) == 2):
        settingsFileName = sys.argv[1]
    else:
        settingsFileName = 'config.ini'
    settings = Settings(settingsFileName)
    sameFolder = False
    if settings.tampered_folder == settings.authentic_folder:
        sameFolder = True

    # create workplace if not available
    if not(os.path.isdir(settings.working_folder)):
        os.makedirs(settings.working_folder)

    # Get db stats
    tp_filelist = glob.glob1(settings.tampered_folder, '*.*')
    au_filelist = glob.glob1(settings.authentic_folder, '*.*')
    n_tp = len(tp_filelist)
    n_au = len(au_filelist)
    print('nb authentic images: {}'.format(n_au))
    print('nb tampered images: {}'.format(n_tp))

    # n_tp = 30
    # n_au = 40
    sys.stdout.flush()
    # authentic list
    if not sameFolder:
        print('Collecting authentic images.')
        au_str_list = []
        au_borderList = []
        for i in tqdm(range(n_au)):
            borderName, thr = getBorderFile(au_filelist[i], settings.borders_authentic_folder)
            au_borderList.append(borderName)
            au_str_list.append('{},{},{},{}'.format(au_filelist[i],
                                                    0,
                                                    borderName,
                                                    'none'))
    else:
        print('Authentic and Tampered are the same folder. Using Tampered only.')

    # tampered list
    print('Collecting tampered images.')
    tp_str_list = []
    tp_borderlist = []
    tp_masklist = []
    for i in tqdm(range(n_tp)):
        borderName, thr = getBorderFile(tp_filelist[i], settings.borders_tampered_folder)
        tp_borderlist.append(borderName)
        maskName, isBlurred = getMaskedFile(tp_filelist[i], settings.mask_tampered_folder)
        tp_masklist.append(maskName)
        # tp_str_list.append('{},{},{},{}'.format(os.path.join(settings.tampered_folder, tp_filelist[i]),
        #                                         1,
        #                                         os.path.join(settings.borders_tampered_folder, borderName),
        #                                         os.path.join(settings.mask_tampered_folder, maskName)))
        tp_str_list.append('{},{},{},{}'.format(tp_filelist[i],
                                                1,
                                                borderName,
                                                maskName))

    n_per_class = int(round(n_tp * settings.pct_test))
    # for each test file
    for tset in range(settings.kfold):
        print('Creating testset {}'.format(tset))
        shuffle(tp_str_list)
        tp_test = tp_str_list[0:n_per_class]
        tp_training = tp_str_list[n_per_class:]

        if not sameFolder:
            shuffle(au_str_list)
            au_test = au_str_list[0:n_per_class]
            au_training = au_str_list[n_per_class:]
            test = au_test + tp_test
            training = au_training + tp_training
        else:
            test = tp_test
            training = tp_training

        shuffle(test)
        shuffle(training)
        trainingfileout = os.path.join(settings.working_folder, 'training_t{}.txt'.format(tset+1))
        testfileout = os.path.join(settings.working_folder, 'test_t{}.txt'.format(tset+1))
        printFileList(trainingfileout, training)
        printFileList(testfileout, test)


if __name__ == '__main__':
    main()