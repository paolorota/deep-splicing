import os
import datetime
import configparser
import tensorflow as tf
import layers as L
import numpy as np
import time
from script_siamese_data_creation import DataManager_siamese


class Settings:
    def __init__(self, filename):
        print('Reading params form ' + filename)
        config = configparser.ConfigParser()
        config.read(filename)
        self.logdir = config.get('General', 'logdir')
        self.datapath = config.get('Train', 'data_file_path')


class Main:
    def __init__(self, sets):
        self.sets = sets
        self.size_image = 80
        self.n_channels = 3
        self.batch_size = 32
        self.learning_rate = 0.0001
        self.epochs = 500

        # net parameters
        self.logits = None
        self.cost = None
        self.accuracy = None
        self.optimizer = None
        self.norm_probe = None
        self.filters = None


    def network_linear(self, x1, x2):
        with tf.name_scope('Tower_1'):
            net1 = L.flatten(x1)
            net1 = L.linear(net1, 1024, 'lin1_1')
            net1 = tf.nn.relu(net1)
            net1 = L.linear(net1, 512, 'lin1_2')
            net1 = tf.nn.relu(net1)
            net1 = L.linear(net1, 128, 'lin1_3')
            net1 = tf.nn.relu(net1)
        with tf.name_scope('Tower_2'):
            net2 = L.flatten(x2)
            net2 = L.linear(net2, 1024, 'lin2_1')
            net2 = tf.nn.relu(net2)
            net2 = L.linear(net2, 512, 'lin2_2')
            net2 = tf.nn.relu(net2)
            net2 = L.linear(net2, 128, 'lin2_3')
            net2 = tf.nn.relu(net2)
        with tf.name_scope('concat'):
            netfused = tf.concat(values=[net1, net2], concat_dim=1)
        with tf.name_scope('out'):
            out = L.linear(netfused, 2, 'lin_out')
            # out = tf.nn.relu(out)
        print('Logit tensor: {}'.format(out.get_shape()))
        return out

    def main(self):

        launch_time = datetime.datetime.now()
        log = 'sia_{0:4}{1:2}{2:2}_{3:2}{4:2}{5:2}'.format(launch_time.year,
                                                           launch_time.month,
                                                           launch_time.day,
                                                           launch_time.hour,
                                                           launch_time.minute,
                                                           launch_time.second).replace(' ', '0')
        log_dir = os.path.join(self.sets.logdir, log)
        data_reader = DataManager_siamese()
        data_reader.read_from_h5(self.sets.datapath)
        data_reader.y = data_reader.to_categorical()
        x1_val, x2_val, y_val = data_reader.fetch_random_n_samples_x_class(40)
        print('creating the net')
        x1 = tf.placeholder(tf.float32,
                            shape=[None, data_reader.x1.shape[1], data_reader.x1.shape[2], data_reader.x1.shape[3]],
                            name='x1_input')
        x2 = tf.placeholder(tf.float32,
                            shape=[None, data_reader.x2.shape[1], data_reader.x2.shape[2], data_reader.x2.shape[3]],
                            name='x2_input')
        y = tf.placeholder(tf.float32,
                           shape=[None, 2],
                           name='y_output')
        lr = tf.placeholder(tf.float32)
        # TODO: check if we need a phase here (batch norm)
        phase = tf.placeholder(tf.bool, name='phase')

        print('Input tensor 1: {}'.format(x1.get_shape()))
        print('Input tensor 2: {}'.format(x2.get_shape()))
        print('labels tensor: {}'.format(y.get_shape()))
        self.logits = self.network_linear(x1, x2)
        with tf.name_scope('Crossentropy'):
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=y))
        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(y, 1))
        with tf.name_scope('Optimizer'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss=self.cost)
        with tf.name_scope('Accuracy'):
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar('loss', self.cost)
        tf.summary.scalar('lr', lr)
        tf.summary.scalar('accuracy', self.accuracy)
        merged_summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph(), flush_secs=10)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            print('Entering the session.')
            sess.run(init)
            for epoch in range(0, self.epochs):
                te = time.time()
                mylr = self.learning_rate + epoch * self.learning_rate
                if mylr > 0.1:
                    mylr = 0.1
                print('+*+* Starting Epoch {}'.format(data_reader.epoch))
                for it in range(0, data_reader.x1.shape[0], self.batch_size):
                    # fetch batch and labels
                    batch_imgs1, batch_imgs2, batch_y = data_reader.next_batch(self.batch_size)
                    # run optimization
                    _, cost = sess.run([self.optimizer, self.cost], feed_dict={x1: batch_imgs1,
                                                                               x2: batch_imgs2,
                                                                               y: batch_y,
                                                                               phase: 1,
                                                                               lr: mylr})

                val_loss, val_acc, summary = sess.run([self.cost, self.accuracy, merged_summary_op],
                                                      feed_dict={x1: x1_val,
                                                                 x2: x2_val,
                                                                 y: y_val,
                                                                 phase: 0,
                                                                 lr: mylr})
                summary_writer.add_summary(summary, epoch)
                print('Accuracy for epoch {}: {} - loss: {}'.format(epoch, val_acc, val_loss))
                # print('Accuracy for epoch (different validation)  {}: {}'.format(epoch, val_loss1))
                print('Time for epoch: {}'.format(time.time() - te))






if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='list of possible arguments')
    parser.add_argument('-s', type=str, metavar='config', help="config file input", default='config.cfg')
    args = parser.parse_args()
    set = Settings(args.s)
    m = Main(set)
    m.main()
