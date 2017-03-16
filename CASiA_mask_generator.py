import os
import sys
import glob
import configparser
from scipy import misc
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pylab import get_current_fig_manager
from matplotlib.widgets import Slider, Button, RadioButtons
# from tkinter import *


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
        self.kernel = int(Config.get('Dataset', 'Kernel_size'))
        self.threshold = int(Config.get('Dataset', 'Image_threshold'))


class Stuff:
    def __init__(self, settingsFileName):
        print('Init :)')
        self.img_out = np.zeros((1, 1))
        self.img_diff = np.zeros((1, 1))
        self.img_thr = np.zeros((1, 1))
        self.settings = Settings(settingsFileName)
        self.long_file_name = ''

    def PreliminaryCheck(self, dir_mask, dir_tampered):
        '''
        Used to check the files that have been done already in the mask directory
        :param dir_mask: directory of mask files
        :param dir_tampered: directory of tampered files
        :return: List of tampered files left to do
        '''
        file_list_mask = glob.glob1(dir_mask, '*.png')
        tp_files = glob.glob1(dir_tampered, '*')
        for i, fname in enumerate(file_list_mask):
            nname, ext = os.path.splitext(fname)
            nname = nname.replace('_mask', '')
            tp_files = [x for x in tp_files if nname not in x]
        return tp_files

    def TamperedParser(self, file):
        '''
        Parses the tampered file in order to retrieve the authentic images that are composing it
        :param file: Tampered file name
        :param Au_folder: Authentic images folder
        :return: Two composing images (in gray levels)
        '''
        base = os.path.basename(file)
        sp_list = base.split('_')
        img_bg_name = sp_list[-3]
        img_fg_name = sp_list[-2]
        im_list_bg = glob.glob(os.path.join(self.settings.authentic_folder, 'Au_{}_{}.*'.format(img_bg_name[:3], img_bg_name[3:])))

        if len(im_list_bg) == 1:
            img_bg = misc.imread(im_list_bg[0], flatten=True)
        elif len(im_list_bg) > 1:
            raise Exception('I cannot have more than one file with this root: {}'.format(base))
        else:
            raise Exception('It seems that {} is composed by missing Authentic images.'.format(base))
        im_list_fg = glob.glob(os.path.join(self.settings.authentic_folder, 'Au_{}_{}.*'.format(img_fg_name[:3], img_fg_name[3:])))

        if len(im_list_fg) == 1:
            img_fg = misc.imread(im_list_fg[0], flatten=True)
        elif len(im_list_bg) > 1:
            raise Exception('I cannot have more than one file with this root: {}'.format(base))
        else:
            raise Exception('It seems that {} is composed by missing Authentic images.'.format(base))
        return img_bg, img_fg

    def save(self, event):
        '''
        Button save event
        :param event:
        :return:
        '''
        file_out = os.path.basename(self.long_file_name)
        file_out, ext = os.path.splitext(file_out)
        file_out = os.path.join(self.settings.mask_tampered_folder, '{}_mask.png'.format(file_out))
        if not os.path.exists(self.settings.mask_tampered_folder):
            os.mkdir(self.settings.mask_tampered_folder)
        cv2.imwrite(file_out, self.img_out)
        print('saved in {}'.format(file_out))
        plt.close()

    def update(self, val):
        '''
        On the update of the sliders it updates the pictures
        :param val:
        :return:
        '''
        thr = int(val)
        ker = int(val)
        ret, self.img_thr = cv2.threshold(self.img_diff, thr, 255, cv2.THRESH_BINARY)
        kernel = np.ones((ker, ker))
        self.img_out = cv2.morphologyEx(self.img_thr, cv2.MORPH_CLOSE, kernel)
        # img_close = cv2.erode(img_close, kernel, iterations=1)

        connectivity = 8
        self.img_out = self.img_out.astype(np.uint8)
        labels = np.zeros((1, 1))
        stats = np.zeros((1, 1))
        centroids = np.zeros((1, 1))
        output = cv2.connectedComponentsWithStats(self.img_out, labels=labels,
                                                  stats=cv2.CC_STAT_LEFT,
                                                  centroids=centroids,
                                                  connectivity=connectivity)
        # Image Threshold
        plt.subplot(335)
        plt.imshow(self.img_thr, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.title('Image threshold')
        # Image filtered
        plt.subplot(338)
        plt.imshow(self.img_out, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.title('Image after morphological')
        # Image blobs
        plt.subplot(339)
        plt.imshow(output[1])
        plt.xticks([])
        plt.yticks([])
        plt.title('CC detection')


    def main(self):
        tp_files = self.PreliminaryCheck(self.settings.mask_tampered_folder,
                                         self.settings.tampered_folder)

        for i, file_name in enumerate(tp_files):
            self.long_file_name = os.path.join(self.settings.tampered_folder,
                                               file_name)
            print('Working on: {}'.format(self.long_file_name))
            img_bg, img_fg = self.TamperedParser(self.long_file_name)

            img_tp = misc.imread(self.long_file_name, flatten=True)

            self.img_diff = np.floor(np.abs(img_tp - img_bg))

            ret, self.img_thr = cv2.threshold(self.img_diff, self.settings.threshold, 255, cv2.THRESH_BINARY)
            kernel = np.ones((self.settings.kernel, self.settings.kernel))
            self.img_out = cv2.morphologyEx(self.img_thr, cv2.MORPH_CLOSE, kernel)
            # img_close = cv2.erode(img_close, kernel, iterations=1)

            connectivity = 8
            self.img_out = self.img_out.astype(np.uint8)
            labels = np.zeros((1, 1))
            stats = np.zeros((1, 1))
            centroids = np.zeros((1, 1))
            output = cv2.connectedComponentsWithStats(self.img_out, labels=labels,
                                                      stats=cv2.CC_STAT_LEFT,
                                                      centroids=centroids,
                                                      connectivity=connectivity)
            # Get the results
            # The first cell is the number of labels
            num_labels = output[0]
            print('Num labels: {}'.format(num_labels))
            # The second cell is the label matrix
            labels = output[1]
            # # The third cell is the stat matrix
            # stats = output[2]
            # # The fourth cell is the centroid matrix
            # centroids = output[3]

            ##################
            #
            #  PLOT STUFF!!!
            #
            ##################
            fig, ax = plt.subplots(figsize=(10, 7))
            mgr = plt.get_current_fig_manager()
            mgr.window.wm_geometry("+100+20")
            plt.subplots_adjust(bottom=0.25)
            # Image 1
            plt.subplot(331)
            plt.imshow(img_bg, cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.title('Image 1')
            # Image 2
            plt.subplot(332)
            plt.imshow(img_fg, cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.title('Image 2')
            # Image tampered
            plt.subplot(333)
            plt.imshow(misc.imread(self.long_file_name))
            plt.xticks([])
            plt.yticks([])
            plt.title('Image Tampered')
            # Image difference
            plt.subplot(334)
            plt.imshow(self.img_diff, cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.title('Image difference')
            # Image Threshold
            plt.subplot(335)
            plt.imshow(self.img_thr, cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.title('Image threshold')
            # Image filtered
            plt.subplot(336)
            plt.imshow(self.img_out, cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.title('Image after morphological')
            # Image blobs
            plt.subplot(337)
            plt.imshow(labels)
            plt.xticks([])
            plt.yticks([])
            plt.title('CC detection')

            axthr = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='pink')
            axker = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor='pink')
            sthr = Slider(axthr, 'Thr', 1, 40, valinit=self.settings.threshold)
            sker = Slider(axker, 'Kernel', 1, 15, valinit=self.settings.kernel)

            sthr.on_changed(self.update)
            sker.on_changed(self.update)

            saveax = plt.axes([0.78, 0.03, 0.12, 0.05], facecolor='pink')
            buttonSave = Button(saveax, 'Save and close', color='b', hovercolor='pink')
            buttonSave.on_clicked(self.save)
            # exitax = plt.axes([0.60, 0.03, 0.12, 0.05], facecolor='pink')
            # buttoExit = Button(exitax, 'Exit', color='g', hovercolor='pink')
            # buttoExit.on_clicked(self.exit)
            plt.show()


if __name__ == '__main__':
    # Params
    if len(sys.argv) == 2:
        settingsFileName = sys.argv[1]
    else:
        settingsFileName = 'config.ini'
    a = Stuff(settingsFileName)
    a.main()
