import h5py
import time
import tensorflow as tf
import os
import datetime
from os.path import join
from dcgan import DCGAN
import shutil


h5_file = '/home/prota/Datasets/utzap.h5'
logdir = './log'
sampledir = './samples'
expname = 'test_di_prova'

def readH5(h5file):
    with h5py.File(h5file, 'r') as f:
        x = f['train_canny'].value
        y = f['train_img'].value
    return x, y

launch_time = datetime.datetime.now()
# log = 'fgan_y{0:4}_m{1:2}_d{2:2}_h{3:2}{4:2}{5:2}'.format(launch_time.year,
#                                                           launch_time.month,
#                                                           launch_time.day,
#                                                           launch_time.hour,
#                                                           launch_time.minute,
#                                                           launch_time.second).replace(' ', '0')
# this_logdir = join(logdir, log)
# if not os.path.exists(logdir):
#     os.mkdir(logdir)
# sample_dir = join('samples', log)
# if not os.path.exists(sample_dir):
#     os.mkdir(sample_dir)

this_logdir = join(logdir, expname)
if os.path.exists(this_logdir):
    shutil.rmtree(this_logdir)
os.mkdir(this_logdir)

this_sampledir = join(sampledir, expname)
if os.path.exists(this_sampledir):
    shutil.rmtree(this_sampledir)
os.mkdir(this_sampledir)


t0 = time.time()
x, y = readH5(h5_file)

# data normalization
x = x / 255
y = y / 255

print('shape x: {}'.format(x.shape))
print('shape y: {}'.format(y.shape))
print('data loaded in {0:3.3} sec'.format(time.time() - t0))

gan = DCGAN(x, y, log_dir=this_logdir, sample_dir=this_sampledir)
print('creating the model of the network.')
gan.build_gan()
print('gan model prepared.')
with tf.Session() as sess:
    gan.train_gan(sess=sess)
