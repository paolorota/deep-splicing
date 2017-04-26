import tensorflow as tf
import layers as L

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./data/mnist", one_hot=True)

# Parameters
learning_rate = 0.001
training_iters = 200000
batch_size = 128
display_step = 50
dropout_prob = 0.75 # Dropout, probability to keep units

train_shape = mnist.train._images.shape
test_shape = mnist.test._images.shape
n_classes = mnist.train._labels.shape[-1]

def conv(tensor, W, b, strides=1):
    tensor = tf.nn.conv2d(tensor, W, strides=[1, strides, strides, 1], padding='SAME')
    tensor = tf.nn.bias_add(tensor, b)
    return tf.nn.relu(tensor)

def pool(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

weights = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    'fc1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    'fc2': tf.Variable(tf.random_normal([1024, n_classes]))
}
biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bf1': tf.Variable(tf.random_normal([1024])),
    'bf2': tf.Variable(tf.random_normal([n_classes])),
}

# create convenet
x = tf.placeholder(tf.float32, [None, train_shape[-1]])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

reshape_layer = tf.reshape(x, shape=[-1, 28, 28, 1])
conv1 = conv(reshape_layer, W=weights['wc1'], b=biases['bc1'])
pool1 = pool(conv1, 2)

# conv2 = conv(pool1, W=weights['wc2'], b=biases['bc2'])
conv2 = L.conv(pool1, 'conv1', 5, 5, 64)
pool2 = pool(conv2, 2)

flat1 = L.flatten(pool2)
#1
# fc1 = L.linear(flat1, 1024, 'linear1')
#2
fc1 = tf.add(tf.matmul(flat1, weights['fc1']), biases['bf1'])
#3
# with tf.variable_scope('linear1'):
#     shape = flat1.get_shape().as_list()
#     matrix = tf.get_variable("w", [shape[1], 1024], tf.float32, tf.truncated_normal_initializer())
#     bias = tf.get_variable('b', [1024], initializer=tf.constant_initializer(0.0))
#     fc1 = tf.nn.bias_add(tf.matmul(flat1, matrix), bias=bias)
#4
# with tf.variable_scope('fc') as scope:
#     # use weight of dimension 7 * 7 * 64 x 1024
#     input_features = shape = flat1.get_shape().as_list()[1]
#     w = tf.get_variable('weights', [input_features, 1024],
#                         initializer=tf.truncated_normal_initializer())
#     b = tf.get_variable('biases', [1024],
#                         initializer=tf.constant_initializer(0.0))
#     fc1 = tf.nn.bias_add(tf.matmul(flat1, w), b)

fc1 = tf.nn.relu(fc1)
fc1 = tf.nn.dropout(fc1, keep_prob)
pred = tf.add(tf.matmul(fc1, weights['fc2']), biases['bf2'])


# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# write summaries
tf.summary.scalar('Loss', cost)
tf.summary.scalar('Accuracy', accuracy)
tf.summary.image('input', reshape_layer, max_outputs=3)
tf.summary.histogram('filters_1', conv1)
tf.summary.histogram('filters_2', conv2)
merged_summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(logdir='./log/mnist', graph=tf.get_default_graph(), flush_secs=5)

# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout_prob})
        if step % display_step == 0:
            loss, acc, summary = sess.run([cost, accuracy, merged_summary_op],
                                          feed_dict={x: batch_x,
                                                     y: batch_y,
                                                     keep_prob: dropout_prob})
            summary_writer.add_summary(summary, step)
            print('Batch Iter: {} --> loss: {} --> acc: {}'.format(step, loss, acc))
        step += 1
