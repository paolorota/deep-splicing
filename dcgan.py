from layers import *
from six.moves import xrange
import scipy


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


class DCGAN:
    def __init__(self, x, y, log_dir='./log', sample_dir='./samples'):
        self.size_image = 128
        self.batch_size = 32
        self.learning_rate = 0.0001
        self.epochs = 400
        self.sample_num = 32

        self.n_channelsX = 1
        self.n_channelsY = 3

        self.gf_dim = 64
        self.gfc_dim = 1024
        self.c_dim = 3  # output of the generator
        self.df_dim = 64
        self.z_dim = 100

        self.sample_dir = sample_dir
        self.log_dir = log_dir
        self.train_x = x
        self.train_y = y

    def generator(self, x, ngf=64, n_layers=7):
        with tf.variable_scope("generator") as scope:
            size_layers = []
            for i in range(n_layers):
                if i == 0:
                    size_layers.append(self.size_image)
                else:
                    size_layers.append(int(size_layers[-1] / 2))
            # s_h, s_w = self.size_image, self.size_image
            # s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)  # 64
            # s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)  # 32
            # s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)  # 16
            # s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)  # 8
            # s_h32, s_w32 = conv_out_size_same(s_h16, 2), conv_out_size_same(s_w16, 2)  # 4
            # s_h64, s_w64 = conv_out_size_same(s_h32, 2), conv_out_size_same(s_w32, 2)  # 2

            layers = []
            # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
            with tf.variable_scope("encoder_1"):
                output = conv2d(x, ngf, d_h=2, d_w=2, name='g_h1_enc')
                layers.append(output)

            layer_specs = [
                ngf * 2,  # encoder_2: [batch, 64, 64, ngf] => [batch, 32, 32, ngf * 2]
                ngf * 4,  # encoder_3: [batch, 32, 32, ngf * 2] => [batch, 16, 16, ngf * 4]
                ngf * 8,  # encoder_4: [batch, 16, 16, ngf * 4] => [batch, 8, 8, ngf * 8]
                ngf * 8,  # encoder_5: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
                ngf * 8,  # encoder_6: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
                ngf * 8,  # encoder_7: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
            ]

            for i, out_channels in enumerate(layer_specs):
                with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
                    rectified = lrelu(layers[-1], 0.2)
                    # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
                    convolved = conv2d(rectified, out_channels, d_w=2, d_h=2, name='g_h{}_enc'.format(i))
                    output = batchnorm(convolved)
                    layers.append(output)

            layer_specs = [
                (ngf * 8, 0.5),  # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
                (ngf * 8, 0.5),  # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
                (ngf * 8, 0.5),  # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
                (ngf * 4, 0.0),  # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
                (ngf * 2, 0.0),  # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
                (ngf, 0.0)  # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
            ]

            num_encoder_layers = len(layers)
            for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
                skip_layer = num_encoder_layers - decoder_layer - 1
                with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
                    if decoder_layer == 0:
                        # first decoder layer doesn't have skip connections
                        # since it is directly connected to the skip_layer
                        input = layers[-1]
                    else:
                        input = tf.concat([layers[-1], layers[skip_layer]], axis=3)
                    rectified = tf.nn.relu(input)
                    output_shape = [self.batch_size,
                                    size_layers[-1 - decoder_layer],
                                    size_layers[-1 - decoder_layer],
                                    out_channels]
                    output =deconv2d(rectified, output_shape, k_h=4, k_w=4,
                                     name='g_h{}_deconv'.format(len(layer_specs) - decoder_layer))
                    output = batchnorm(output)
                    if dropout > 0.0:
                        output = tf.nn.dropout(output, keep_prob=1 - dropout)
                    layers.append(output)
            # decoder_1: [batch, 64, 64, ngf] => [batch, 128, 128, generator_outputs_channels]
            with tf.variable_scope("decoder_1"):
                input = tf.concat([layers[-1], layers[0]], axis=3)
                rectified = tf.nn.relu(input)
                output_shape = [self.batch_size,
                                self.size_image,
                                self.size_image,
                                self.c_dim]
                output = deconv2d(rectified, output_shape, k_h=4, k_w=4, name='g_h1_dec')
                output = tf.tanh(output)
                layers.append(output)
            return layers[-1]

    def sampler(self, x, ngf=64, n_layers=7):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()
            size_layers = []
            for i in range(n_layers):
                if i == 0:
                    size_layers.append(self.size_image)
                else:
                    size_layers.append(int(size_layers[-1] / 2))
            # s_h, s_w = self.size_image, self.size_image
            # s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)  # 64
            # s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)  # 32
            # s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)  # 16
            # s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)  # 8
            # s_h32, s_w32 = conv_out_size_same(s_h16, 2), conv_out_size_same(s_w16, 2)  # 4
            # s_h64, s_w64 = conv_out_size_same(s_h32, 2), conv_out_size_same(s_w32, 2)  # 2

            layers = []
            # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
            with tf.variable_scope("encoder_1"):
                output = conv2d(x, ngf, d_h=2, d_w=2, name='g_h1_enc')
                layers.append(output)

            layer_specs = [
                ngf * 2,  # encoder_2: [batch, 64, 64, ngf] => [batch, 32, 32, ngf * 2]
                ngf * 4,  # encoder_3: [batch, 32, 32, ngf * 2] => [batch, 16, 16, ngf * 4]
                ngf * 8,  # encoder_4: [batch, 16, 16, ngf * 4] => [batch, 8, 8, ngf * 8]
                ngf * 8,  # encoder_5: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
                ngf * 8,  # encoder_6: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
                ngf * 8,  # encoder_7: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
            ]

            for i, out_channels in enumerate(layer_specs):
                with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
                    rectified = lrelu(layers[-1], 0.2)
                    # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
                    convolved = conv2d(rectified, out_channels, d_w=2, d_h=2, name='g_h{}_enc'.format(i))
                    output = batchnorm(convolved)
                    layers.append(output)

            layer_specs = [
                (ngf * 8, 0.5),  # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
                (ngf * 8, 0.5),  # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
                (ngf * 8, 0.5),  # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
                (ngf * 4, 0.0),  # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
                (ngf * 2, 0.0),  # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
                (ngf, 0.0)  # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
            ]

            num_encoder_layers = len(layers)
            for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
                skip_layer = num_encoder_layers - decoder_layer - 1
                with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
                    if decoder_layer == 0:
                        # first decoder layer doesn't have skip connections
                        # since it is directly connected to the skip_layer
                        input = layers[-1]
                    else:
                        input = tf.concat([layers[-1], layers[skip_layer]], axis=3)
                    rectified = tf.nn.relu(input)
                    output_shape = [self.batch_size,
                                    size_layers[-1 - decoder_layer],
                                    size_layers[-1 - decoder_layer],
                                    out_channels]
                    output = deconv2d(rectified, output_shape, k_h=4, k_w=4,
                                      name='g_h{}_deconv'.format(len(layer_specs) - decoder_layer))
                    output = batchnorm(output)
                    if dropout > 0.0:
                        output = tf.nn.dropout(output, keep_prob=1 - dropout)
                    layers.append(output)
            # decoder_1: [batch, 64, 64, ngf] => [batch, 128, 128, generator_outputs_channels]
            with tf.variable_scope("decoder_1"):
                input = tf.concat([layers[-1], layers[0]], axis=3)
                rectified = tf.nn.relu(input)
                output_shape = [self.batch_size,
                                self.size_image,
                                self.size_image,
                                self.c_dim]
                output = deconv2d(rectified, output_shape, k_h=4, k_w=4, name='g_h1_dec')
                output = tf.tanh(output)
                layers.append(output)
            return layers[-1]

    def siamese_tower(self, image, n_layers=5, reuse=False):
        with tf.variable_scope("siamese_tower") as scope:
            if reuse:
                scope.reuse_variables()
            layers = []
            with tf.variable_scope("slayer_1"):
                layers.append(lrelu(conv2d(image, self.df_dim, name='d_tow0_conv')))

            for i in range(2, n_layers + 1):
                with tf.variable_scope("slayer_%d" % i):
                    layers.append(lrelu(batchnorm(conv2d(layers[-1], self.df_dim * (i - 1) * 2, name='d_tow%d_con' % i))))

            layers.append(linear(tf.reshape(layers[-1], [self.batch_size, -1]), 128, 'd_tow_lin'))
            return layers[-1]

    def siamese_discriminator(self, image_in, image_out, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            if image_in.get_shape()[-1] == 3:
                image_in = lrelu(conv2d(image_in, 1, d_h=1, d_w=1, name='d_hprel_in'))
            if image_out.get_shape()[-1] == 3:
                image_out = lrelu(conv2d(image_out, 1, d_h=1, d_w=1, name='d_hprel_out'))
            t1 = self.siamese_tower(image_in, reuse=reuse)
            t2 = self.siamese_tower(image_out, reuse=True)
            t3 = concat([t1, t2], 1)
            t4 = linear(t3, 64, 'd_siam1_lin')
            t5 = linear(t4, 1, 'd_siam2_lin')
            return tf.nn.sigmoid(t5), t5

    def create_discriminator(self, discrim_inputs, discrim_targets, reuse=False):

        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            n_layers = 3
            layers = []

            # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]

            input = tf.concat([discrim_inputs, discrim_targets], axis=3)

            # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
            with tf.variable_scope("layer_1"):
                convolved = conv2d(input_=input, output_dim=self.df_dim, k_h=4, k_w=4)
                rectified = lrelu(convolved, 0.2)
                layers.append(rectified)

            # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
            # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
            # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
            for i in range(n_layers):
                with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                    out_channels = self.df_dim * min(2 ** (i + 1), 8)
                    stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
                    convolved = conv2d(input_=layers[-1], output_dim=out_channels, k_h=4, k_w=4, d_h=stride, d_w=stride)
                    normalized = batchnorm(convolved)
                    rectified = lrelu(normalized, 0.2)
                    layers.append(rectified)

            # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                convolved = conv2d(rectified, output_dim=1, d_h=1, d_w=1)
                output = tf.sigmoid(convolved)
                layers.append(output)

            return layers[-1], convolved

    def build_gan(self):
        # TODO: fix naming of variables

        self.x = tf.placeholder(tf.float32,
                                shape=[self.batch_size, self.size_image, self.size_image, self.n_channelsX],
                                name='img_input')
        self.y = tf.placeholder(tf.float32,
                                shape=[self.batch_size, self.size_image, self.size_image, self.n_channelsY],
                                name='mask_input')
        # self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name='z_input')
        # TODO: maybe use the learning rate!
        self.lr = tf.placeholder(tf.float32, name='learning_rate')

        self.x_sum = tf.summary.image("x", self.x)
        self.y_sum = tf.summary.image("y", self.y)

        self.G = self.generator(self.x)
        self.D, self.D_logits = self.create_discriminator(self.x, self.y, reuse=False)
        self.D_, self.D_logits_ = self.create_discriminator(self.x, self.G, reuse=True)
        # self.D, self.D_logits = self.siamese_discriminator(self.x, self.y, reuse=False)
        # self.D_, self.D_logits_ = self.siamese_discriminator(self.x, self.G, reuse=True)
        self.S = self.sampler(self.x)

        # print('Noise tensor {}'.format(self.z.get_shape()))
        print('Image tensor {}'.format(self.x.get_shape()))
        print('Mask tensor {}'.format(self.y.get_shape()))
        self.d_sum = tf.summary.histogram("d", self.D)
        self.d__sum = tf.summary.histogram("d_", self.D_)
        self.G_sum = tf.summary.image("G", self.G)

        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_)))

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = tf.summary.scalar('g_loss', self.g_loss)
        self.d_loss_sum = tf.summary.scalar('d_loss', self.d_loss)

        # select the variables to train and where
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if var.name.startswith("discriminator")]
        self.g_vars = [var for var in t_vars if var.name.startswith("generator")]

    def train_gan(self, sess):
        d_optim = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(self.g_loss, var_list=self.g_vars)
        tf.global_variables_initializer().run()

        # put together summaries
        self.g_sum = tf.summary.merge([
            self.g_loss_sum, self.G_sum, self.d__sum
        ])
        self.d_sum = tf.summary.merge([
            self.d_loss_sum, self.d_sum
        ])
        self.in_sum = tf.summary.merge([
            self.x_sum, self.y_sum
        ])
        self.writer = tf.summary.FileWriter(self.log_dir, sess.graph)

        # sampling random noise
        # sample_z = np.random.uniform(-1, 1, size=(self.sample_num, self.z_dim))

        # sample inputs and labels
        sample_inputs = self.train_x[0:self.sample_num]
        sample_labels = self.train_y[0:self.sample_num]

        counter = 1
        for epoch in xrange(self.epochs):
            # print('Starting epoch: {}'.format(epoch + 1))
            batch_idxs = len(self.train_x) // self.batch_size

            for idx in xrange(0, batch_idxs):
                batch_images = self.train_x[idx * self.batch_size:(idx + 1) * self.batch_size]
                batch_labels = self.train_y[idx * self.batch_size:(idx + 1) * self.batch_size]
                # batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

                # Update D network
                _, summary_str, summary_str1 = sess.run([d_optim, self.d_sum, self.in_sum],
                                                        feed_dict={
                                                            self.x: batch_images,
                                                            self.y: batch_labels
                                                        })
                # step = (np.float32(epoch) + 1)+(np.float32(idx)/np.float32(batch_idxs))
                self.writer.add_summary(summary=summary_str, global_step=counter)
                self.writer.add_summary(summary=summary_str1, global_step=counter)

                # Update G (twice because it is suggested like so)
                for j in xrange(1):
                    _, summary_str = sess.run([g_optim, self.g_sum],
                                              feed_dict={
                                                  self.x: batch_images,
                                                  self.y: batch_labels
                                              })
                self.writer.add_summary(summary=summary_str, global_step=counter)

                # compute error on training
                errD_fake = self.d_loss_fake.eval({
                    self.x: batch_images,
                    self.y: batch_labels
                })
                errD_real = self.d_loss_real.eval({
                    self.x: batch_images,
                    self.y: batch_labels
                })
                errG = self.g_loss.eval({
                    self.x: batch_images,
                    self.y: batch_labels
                })
                # counter += 1
                print('Epoch [{0}] [{1}/{2}] d_loss: {3:.8} g_loss: {4:.8}'.format(
                    epoch, idx, batch_idxs, errD_fake + errD_real, errG
                ))

                # save images every 100 batch iters
                if np.mod(counter, 100) == 1:
                    samples, d_loss, g_loss = sess.run(
                        [self.S, self.d_loss, self.g_loss],
                        feed_dict={
                            self.x: sample_inputs,
                            self.y: sample_labels
                        }
                    )
                    manifold_h = int(np.ceil(np.sqrt(samples.shape[0])))
                    manifold_w = int(np.ceil(np.sqrt(samples.shape[0])))
                    save_images(samples, [manifold_h, manifold_w], '{}/train_{:02d}_{:04d}_Gx.png'.format(
                        self.sample_dir, epoch, idx))
                    save_images(sample_labels, [manifold_h, manifold_w], '{}/train_{:02d}_{:04d}_y.png'.format(
                        self.sample_dir, epoch, idx))
                    save_images(sample_inputs, [manifold_h, manifold_w], '{}/train_{:02d}_{:04d}_x.png'.format(
                        self.sample_dir, epoch, idx))
                    print("[Sample] d_loss: {0:.8}, g_loss: {1:.8}".format(d_loss, g_loss))
                counter += 1


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if images.shape[3] in (3,4):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3] == 1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter '
                         'must have dimensions: HxW or HxWx3 or HxWx4')


def save_images(images, size, image_path):
    return imsave(images, size, image_path)


def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    from scipy import misc
    return misc.imsave(path, image)