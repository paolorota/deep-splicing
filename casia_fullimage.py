import ConfigParser
import os, sys, glob
import random
import numpy as np
import time
from method_cnn import run_cnn

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
        # self.pct_test = float(Config.get('Dataset', 'Percent_test')) / 100
        # self.kfold = int(Config.get('Dataset', 'K_fold'))
        self.method = Config.get('Test', 'Method')

class TestImage:
    def __init__(self, text, settings):
        s = text.split(',')
        self.label = int(s[1])
        if self.label == 0:
            self.image_path = os.path.join(settings.authentic_folder, s[0])
            self.mask_image = s[3]
        else:
            self.image_path = os.path.join(settings.tampered_folder, s[0])
            self.mask_image = os.path.join(settings.mask_tampered_folder, s[3])
        self.border_image = os.path.join(settings.borders_authentic_folder, s[2])



def readtestfile(filename, settings):
    if not(os.path.exists(filename)):
        raise("File {} does not exist".format(filename))
    with open(filename, mode='r') as fin:
        content = fin.readlines()
    imagelist = []
    for i in range(len(content)):
        thiscontent = content[i][0:-1]
        imagelist.append(TestImage(thiscontent, settings))
    return imagelist


def dummymethod(training_images, test_images):
    nb_test = len(test_images)
    results = np.zeros((nb_test, 1))
    for i in range(nb_test):
        if round(random.random())> 0.5:
            results[i, 0] = 1
    return results


def main():
    # Params
    if (len(sys.argv) == 2):
        settingsFileName = sys.argv[1]
    else:
        settingsFileName = 'config.ini'
    settings = Settings(settingsFileName)

    # search for training and test files
    training_filelist = glob.glob1(settings.working_folder, 'training*')
    test_filelist = glob.glob1(settings.working_folder, 'test*')

    nb_tests = len(test_filelist)

    cumulative_confmat = np.zeros((2,2))
    for t in range(nb_tests):
        training_images = readtestfile(os.path.join(settings.working_folder, training_filelist[t]), settings)
        test_images = readtestfile(os.path.join(settings.working_folder, test_filelist[t]), settings)

        tinit = time.time()
        # try CNN
        if settings.method == 'CNN':
            print('CNN method')
            results = run_cnn(training_images, test_images)
        else:
            # try dummy
            print('Dummy method')
            results = dummymethod(training_images, test_images)
        tend = time.time()

        # calc confusion matrix
        confmat = np.zeros((2,2))
        for i in range(len(test_images)):
            label = test_images[i].label
            prediction = results[i,0]
            confmat[label, prediction] += 1
        cumulative_confmat += confmat

    # get stats
    true_positive = cumulative_confmat[1, 1]
    true_negative = cumulative_confmat[0, 0]
    false_positive = cumulative_confmat[0, 1]
    false_negative = cumulative_confmat[1, 0]
    accuracy = (true_positive + true_negative)/(cumulative_confmat.sum())
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    fscore = 2 * precision * recall / (precision + recall)

    # print
    print("Training-test time: {} secs".format(tend - tinit))
    print('## Results ##')
    print('True Positive: {}'.format(int(true_positive)))
    print('True Negative: {}'.format(int(true_negative)))
    print('False Positive: {}'.format(int(false_positive)))
    print('False Negative: {}'.format(int(false_negative)))
    print('Accuracy: {}'.format(accuracy))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F-score: {}'.format(fscore))


########### END ################

if __name__ == '__main__':
    main()