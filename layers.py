import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
import numpy as np

def conv(input_tensor, name, kw, kh, n_out, dw=1, dh=1, activation_fn=tf.nn.relu):

    # gets the color depth of the input
    n_in = input_tensor.get_shape()[-1].value

    # open a scope which is good for reusing the weights
    with tf.variable_scope(name):
        weights = tf.get_variable('weights', [kh, kw, n_in, n_out], tf.float32,
                                  initializer=tf.truncated_normal_initializer())
        biases = tf.get_variable('bias', [n_out], tf.float32, initializer=tf.constant_initializer())

        # create the convolution
        conv = tf.nn.conv2d(input_tensor, weights, (1, dh, dw, 1), padding='SAME')

        # perform activation
        activation = activation_fn(tf.nn.bias_add(conv, biases))
        return activation


def pool(input_tensor, name, kh, kw, dh=1, dw=1):
    return tf.nn.max_pool(input_tensor,
                          ksize=[1, kh, kw, 1],
                          strides=[1, dh, dw, 1],
                          padding='VALID',
                          name=name)


def deconv(input_tensor, n_out=64, kh=5, kw=5, dh=1, dw=1, name="deconv2d", with_w=False):

    # gets the color depth of the input
    n_in = input_tensor.get_shape()[-1]
    input_shape = input_tensor.get_shape()
    # from here: https://github.com/tensorflow/tensorflow/issues/833
    batch_size = tf.shape(input_tensor)[0]
    output_shape = tf.stack([batch_size, input_shape[1].value, input_shape[2].value, n_out])

    with tf.variable_scope(name):
        # filter: A 4-D Tensor with the same type as value and shape [height, width, output_channels, in_channels].
        # filter's in_channels dimension must match that of value.
        w = tf.get_variable('weights', [kh, kw, n_out, n_in],
                            initializer=tf.truncated_normal_initializer())
        deconv = tf.nn.conv2d_transpose(input_tensor,
                                        w,
                                        output_shape=output_shape,
                                        strides=[1, dh, dw, 1],
                                        padding='SAME')
        # biases = tf.get_variable('bias', n_out, initializer=tf.constant_initializer(0.0))
        # deconv = tf.nn.bias_add(deconv, biases)
        # reshaping needed to preserve the shape in the net
        deconv = tf.reshape(deconv, shape=output_shape)
        if with_w:
            return deconv, w, biases
        else:
            return deconv


def upsample(input_tensor, ks=2):
    s = input_tensor.get_shape()
    return tf.image.resize_images(input_tensor, size=tf.stack([s[1]*ks, s[2]*ks]))


def unpool(value, name='unpool'):
    """N-dimensional version of the unpooling operation from
    https://www.robots.ox.ac.uk/~vgg/rg/papers/Dosovitskiy_Learning_to_Generate_2015_CVPR_paper.pdf

    :param value: A Tensor of shape [b, d0, d1, ..., dn, ch]
    :return: A Tensor of shape [b, 2*d0, 2*d1, ..., 2*dn, ch]
    """
    with tf.name_scope(name) as scope:
        sh = value.get_shape().as_list()
        dim = len(sh[1:-1])
        out = (tf.reshape(value, [-1] + sh[-dim:]))
        for i in range(dim, 0, -1):
            out = tf.concat(i, [out, tf.zeros_like(out)])
        out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
        out = tf.reshape(out, out_size, name=scope)
        return out


def flatten(input_tensor):
    # gets the color depth of the input
    # n_in = input_tensor.get_shape()[-1].value
    shape = input_tensor.get_shape().as_list()
    dim = np.prod(shape[1:])
    tmp_tensor = tf.reshape(input_tensor, [-1, dim])
    return tmp_tensor


def linear(input_, output_size, name, stddev=1, bias_start=0.1, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(name):
        matrix = tf.get_variable("weights", [shape[1], output_size], tf.float32,
                                 tf.truncated_normal_initializer(stddev=stddev))
        bias = tf.get_variable("biases", [output_size],
                               initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.nn.bias_add(tf.matmul(input_, matrix), bias=bias)
