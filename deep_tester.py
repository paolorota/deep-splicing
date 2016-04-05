import ConfigParser
import os, sys, glob
import random
import numpy as np
import time
from method_cnn import train_cnn, test_cnn
import pandas as pd
import casiaDB_handler as casia


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
        self.method = Config.get('Test', 'method')
        self.patch_size = int(Config.get('Test', 'patch_size'))
        self.patch_stride = int(Config.get('Test', 'patch_stride'))
        self.nb_epochs = int(Config.get('NN', 'nb_epochs'))
        self.batch_size = int(Config.get('NN', 'batch_size'))


class TestImage:
    def __init__(self, text, settings):
        s = text.split(',')
        self.label = int(s[1])
        if self.label == 0:
            self.image_path = os.path.join(settings.authentic_folder, s[0])
            self.mask_image = s[3]
            self.border_image = os.path.join(settings.borders_authentic_folder, s[2])
        else:
            self.image_path = os.path.join(settings.tampered_folder, s[0])
            self.mask_image = os.path.join(settings.mask_tampered_folder, s[3])
            self.border_image = os.path.join(settings.borders_tampered_folder, s[2])


def write_preds(preds, fname):
    pd.DataFrame({"ImageId": list(range(1,len(preds)+1)), "Label": preds}).to_csv(fname, index=False, header=True)


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


def extractStats(confmat):
    # get stats
    true_positive = confmat[1, 1]
    true_negative = confmat[0, 0]
    false_positive = confmat[0, 1]
    false_negative = confmat[1, 0]
    accuracy = (true_positive + true_negative) / (confmat.sum())
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    fscore = 2 * precision * recall / (precision + recall)

    resstring = ['## Stats ##\n']
    resstring.append('True Positive: {}\n'.format(int(true_positive)))
    resstring.append('True Negative: {}'.format(int(true_negative)))
    resstring.append('False Positive: {}'.format(int(false_positive)))
    resstring.append('False Negative: {}'.format(int(false_negative)))
    resstring.append('Accuracy: {}'.format(accuracy))
    resstring.append('Precision: {}'.format(precision))
    resstring.append('Recall: {}'.format(recall))
    resstring.append('F-score: {}'.format(fscore))
    return resstring


def main():
    # Params
    if (len(sys.argv) == 2):
        settingsFileName = sys.argv[1]
    else:
        settingsFileName = 'config.ini'
    settings = Settings(settingsFileName)
    isDebug = 1

    # search for training and test files
    training_filelist = glob.glob1(settings.working_folder, 'training*')
    test_filelist = glob.glob1(settings.working_folder, 'test*')

    nb_tests = len(test_filelist)
    if isDebug == 1:
        nb_tests = 1

    cumulative_confmat = np.zeros((2,2))
    for t in range(nb_tests):
        training_images = readtestfile(os.path.join(settings.working_folder, training_filelist[t]), settings)
        test_images = readtestfile(os.path.join(settings.working_folder, test_filelist[t]), settings)

        # get or create the dataset if not available (if needed)
        tmp_filename_train, tmp_filename_test = casia.create_database(training_images,
                                                                      test_images,
                                                                      prename='tmp',
                                                                      patch_size=settings.patch_size,
                                                                      patch_stride=settings.patch_stride,
                                                                      working_dir=settings.working_folder)
        tinit = time.time()
        # train
        if 'CNN' in settings.method:
            print('Method: {}'.format(settings.method))
            model = train_cnn(tmp_filename_train, tmp_filename_test, settings)
            nb_params = model.count_params()
            results = test_cnn(test_images, model)
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
    statlist = extractStats(cumulative_confmat)

    # Exproting results on file
    results_dir = os.path.join(settings.working_folder, 'results')
    print('Results folder: {}'.format(results_dir))
    try:
        os.stat(results_dir)
    except:
        os.mkdir(results_dir)

    fileout = settings.method + '.DFresults'
    print('Results file: {}'.format(fileout))
    with open(os.path.join(results_dir, fileout), 'w') as f:
        s = []
        s.append('RESULTS for Deep Forensics Splicing detection in images\n')
        s.append('Method: {}\n'.format(settings.method))
        s.append('Patch size: {}\n'.format(settings.patch_size))
        s.append('Patch minimum stride: {}\n'.format(settings.patch_stride))
        if 'CNN' in settings.method:
            s.append('Number of epochs: {}\n'.format(settings.nb_epochs))
            s.append('Batch size: {}\n'.format(settings.batch_size))
            s.append('Number of params in the model: {}\n'.format(nb_params))
            s.append('Model architecture: {}\n'.format(model.to_yaml()))
        f.writelines(s)
        f.writelines(statlist)
        f.write('\n#### RESULTS on single IMAGES####\n')
        json_string = pd.DataFrame({"ImageId": test_images, "Label": results}).to_json()
        f.write(json_string)



########### END ################

if __name__ == '__main__':
    main()