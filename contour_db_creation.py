from os.path import join
import os
import subprocess
import time
import random as rand
import numpy as np

filein = 'list.txt'
wdir = '/home/prota/Datasets/'

curdir = os.curdir
os.chdir(wdir)
with open(join(wdir, filein), 'w') as out:
    subprocess.Popen(['ls', 'ut-zap50k-images/', '-R', '-p'], stdout=out)

time.sleep(1)

with open(join(wdir, filein), 'r') as fin:
    file = fin.read()

lines = file.split('\n')

c_dir = '.'
f_list = list()
for n, line in enumerate(lines):
    if len(line) == 0:
        continue
    if line[-1] == ':':
        c_dir = line[:-1]
        continue

    thisfiles = line.split('\t')
    for f, thisfile in enumerate(thisfiles):
        if thisfile[-1] == '/':
            continue
        else:
            f_list.append(join(c_dir, thisfile))

# generate training and test
test_pct = 0.1

with open(join(wdir, filein), 'w') as fout:
    for i, n in enumerate(f_list):
        fout.write(join(wdir, n) + '\n')


rand.shuffle(f_list)
n_training = int(np.floor(len(f_list) * test_pct))
test_set = f_list[:n_training]
training_set = f_list[n_training:]

with open(join(wdir, 'training.txt'), 'w') as fout:
    for i, n in enumerate(training_set):
        fout.write(join(wdir, n) + '\n')

with open(join(wdir, 'test.txt'), 'w') as fout:
    for i, n in enumerate(test_set):
        fout.write(join(wdir, n) + '\n')
