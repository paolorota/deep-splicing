import os
import sys
from glob import glob
import configparser
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
from skimage import color


class Settings:
    def __init__(self, filename):
        print('Reading params form ' + filename)
        Config = configparser.ConfigParser()
        Config.read(filename)
        self.tampered_folder = Config.get('Dataset', 'DB_folder_tp')
        self.authentic_folder = Config.get('Dataset', 'DB_folder_au')
        self.borders_tampered_folder = Config.get('Dataset', 'DB_folder_tp_borders')
        self.borders_authentic_folder = Config.get('Dataset', 'DB_folder_au_borders')
        self.mask_tampered_folder = Config.get('Dataset', 'DB_folder_tp_mask')
        self.working_folder = Config.get('Dataset', 'Working_folder')
        self.pct_test = float(Config.get('Dataset', 'Percent_test')) / 100
        self.kfold = int(Config.get('Dataset', 'K_fold'))


def TamperedParser(file, Au_folder):
    base = os.path.basename(file)
    sp_list = base.split('_')
    img_bg_name = sp_list[-3]
    img_fg_name = sp_list[-2]
    im_list_bg = glob(os.path.join(Au_folder, 'Au_{}_{}.*'.format(img_bg_name[:3], img_bg_name[3:])))

    if len(im_list_bg) == 1:
        img_bg = misc.imread(im_list_bg[0], flatten=True)
    elif len(im_list_bg) > 1:
        raise Exception('I cannot have more than one file with this root: {}'.format(base))
    else:
        raise Exception('It seems that {} is composed by missing Authentic images.'.format(base))
    im_list_fg = glob(os.path.join(Au_folder, 'Au_{}_{}.*'.format(img_fg_name[:3], img_fg_name[3:])))

    if len(im_list_fg) == 1:
        img_fg = misc.imread(im_list_fg[0], flatten=True)
    elif len(im_list_bg) > 1:
        raise Exception('I cannot have more than one file with this root: {}'.format(base))
    else:
        raise Exception('It seems that {} is composed by missing Authentic images.'.format(base))
    return img_bg, img_fg


def main():
    # Params
    if (len(sys.argv) == 2):
        settingsFileName = sys.argv[1]
    else:
        settingsFileName = 'config.ini'
    settings = Settings(settingsFileName)

    # Listing tampered files
    tp_files = os.listdir(settings.tampered_folder)

    for i, file_name in enumerate(tp_files):
        long_file_name = os.path.join(settings.tampered_folder,
                                      file_name)
        print(long_file_name)
        img_bg, img_fg = TamperedParser(long_file_name, settings.authentic_folder)

        img_tp = misc.imread(long_file_name, flatten=True)

        img_diff = np.floor(np.abs(img_tp - img_bg))
        plt.imshow(img_diff, cmap='gray')
        plt.show()






if __name__ == '__main__':
    main()