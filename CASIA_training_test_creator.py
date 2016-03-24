import time
import os, sys
import glob
import ConfigParser
import myfilelib as my
from tqdm import tqdm


class Settings:
    def __init__(self, filename):
        print('Reading params form ' + filename)
        Config = ConfigParser.ConfigParser()
        Config.read(filename)
        self.tampered_folder = Config.get('Dataset', 'DB_folder_tp')
        self.authentic_folder = Config.get('Dataset', 'DB_folder_au')
        self.borders_tampered_folder = Config.get('Dataset', 'DB_folder_tp_borders')
        self.borders_authentic_folder = Config.get('Dataset', 'DB_folder_au_borders')
        self.working_folder = Config.get('Dataset', 'Working_folder')
        self.pct_test = int(Config.get('Dataset', 'Percent_test'))
        self.kfold = int(Config.get('Dataset', 'K_fold'))

def getBorderFile(filename, dirborder):
    mydir, fname, exp = my.fileparts(filename)
    bordername = '{}_*'.format(fname)
    bordername = glob.glob(os.path.join(dirborder, bordername))[0]
    mydir1, fname1, exp1 = my.fileparts(bordername)
    spl = fname1.split('_t')
    thr = float(spl[1])
    fname1 = '{}.{}'.format(fname1, exp1)
    return fname1, thr


def getMaskedFile(filename, dirmask):
    mydir, fname, exp = my.fileparts(filename)
    maskname = '{}_*'.format(fname)
    maskname = glob.glob(os.path.join(dirmask, maskname))[0]
    mydir1, fname1, exp1 = my.fileparts(maskname)
    spl = fname1.split('_b')
    isBlurred = bool(spl[1])
    fname1 = '{}.{}'.format(fname1, exp1)
    return fname1, isBlurred


def main():
    # Params
    if (len(sys.argv) == 2):
        settingsFileName = sys.argv[1]
    else:
        settingsFileName = 'config.ini'
    settings = Settings(settingsFileName)


    # Get db stats
    tp_filelist = glob.glob1(settings.tampered_folder, '*.*')
    au_filelist = glob.glob1(settings.authentic_folder, '*.*')
    n_tp = len(tp_filelist)
    n_au = len(au_filelist)
    print('nb authentic images: {}'.format(n_au))
    print('nb tampered images: {}'.format(n_tp))

    # authentic list
    au_borderList = []
    for i in tqdm(range(n_au)):
        maskName, thr = getBorderFile(au_filelist[i], settings.borders_authentic_folder)
        au_borderList.append(maskName)


    # tampered list

    # for each test file
    for tset in range(settings.kfold):
        print('Creating testset {}'.format(tset))



if __name__ == '__main__':
    main()