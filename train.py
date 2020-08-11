# !/usr/bin/env python
# title           :train.py
# description     :to train the model
# author          :Deepak Birla
# date            :2018/10/30
# usage           :python train.py --options
# python_version  :3.5.4

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Network import Generator, Discriminator
import Utils_model
import Utils
from Utils_model import VGG_LOSS

from keras.models import Model, load_model
from keras.layers import Input
from tqdm import tqdm
import numpy as np
import argparse
import tensorflow as tf
import datetime
import loadfile
import os

tf.compat.v1.reset_default_graph()

np.random.seed(10)
# Better to use downscale factor as 4 sub-pixel_shape
downscale_factor = 4
# Remember to change image shape if you are having different size of images
# image_shape = (128,128,3)
image_height = 32
image_width = 32
image_h_lr = image_height / 4
image_w_lr = image_width / 4
amount = 6000
batch_size = 2
image_folder = './train2019'
model_path = './model/'

image_shape = (image_height, image_width, 3)
print("==============================")
print(image_shape)
print("==============================")


class TensorboardSummary:
    def __init__(self):
        self.sr_img = tf.placeholder(dtype=tf.float16, shape=[1, image_height, image_width, 3])
        self.hr_img = tf.placeholder(dtype=tf.float16, shape=[1, image_height, image_width, 3])
        self.lr_img = tf.placeholder(dtype=tf.float16, shape=[1, image_h_lr, image_w_lr, 3])
        self.d_loss = tf.placeholder(dtype=tf.float16, shape=None)
        self.gan_loss = tf.placeholder(dtype=tf.float16, shape=None)

        tf.compat.v1.summary.image('HR_Image', self.hr_img, max_outputs=1)
        tf.compat.v1.summary.image('LR_Image', self.lr_img, max_outputs=1)
        tf.compat.v1.summary.image('SR_Image', self.sr_img, max_outputs=1)
        tf.compat.v1.summary.scalar('discriminator_loss', self.d_loss)
        tf.compat.v1.summary.scalar('gan_loss', self.gan_loss)

        self.merged = tf.compat.v1.summary.merge_all()


# Combined network
def get_gan_network(discriminator, shape, generator, optimizer, vgg_loss):
    discriminator.trainable = False
    gan_input = Input(shape=shape)
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=[x, gan_output])
    gan.compile(loss=["mean_squared_error", "binary_crossentropy"], loss_weights=[1., 1e-3], optimizer=optimizer)
    # gan.summary()
    return gan


# default values for all parameters are given, if want different values you can give via commandline
# for more info use $python train.py -h
def train(epochs, batch_size, input_dir, output_dir, model_save_dir, number_of_images, train_test_ratio):
    x_train_lr, x_train_hr, x_test_lr, x_test_hr \
        = Utils.load_training_data(input_dir, '.jpg', number_of_images, train_test_ratio)
    loss = VGG_LOSS(image_shape)
    optimizer = Utils_model.get_optimizer()
    batch_count = int(x_train_hr.shape[0] / batch_size)
    shape = (image_shape[0] // downscale_factor, image_shape[1] // downscale_factor, image_shape[2])

    gen, dis, epoch_start = loadfile.load_dcgan_train_model(model_path, '.h5')

    if os.path.exists(dis) and os.path.exists(gen):

        discriminator = load_model(dis, custom_objects={'vgg_loss': loss.vgg_loss})
        generator = load_model(gen, custom_objects={'vgg_loss': loss.vgg_loss})
        epoch_start = epoch_start + 1

        print("Loaded model... \n With generator {}.............\n With discriminator{}..........\n In epoch{}".format(
            gen, dis, epoch_start))
    else:

        generator = Generator(shape).generator()
        discriminator = Discriminator(image_shape).discriminator()
    generator.compile(loss=loss.vgg_loss, optimizer=optimizer)
    discriminator.compile(loss="binary_crossentropy", optimizer=optimizer)
    gan = get_gan_network(discriminator, shape, generator, optimizer, loss.vgg_loss)

    # loss_file = open(model_save_dir + 'losses.txt', 'w+')

    iteration = 0

    log_dir = "./logs/" + str(datetime.datetime.now())

    init = tf.global_variables_initializer()
    tf_summary = TensorboardSummary()
    loadfile.remove_logs('./logs/')
    with tf.Session() as sess:
        writer = tf.compat.v1.summary.FileWriter(log_dir, sess.graph)

        sess.run(init)

        print('Done writing the summaries')

        for e in range(epoch_start, epochs + 1):
            print('-' * 15, 'Epoch %d' % e, '-' * 15)

            for _ in tqdm(range(batch_count)):
                iteration += 1
                rand_nums = np.random.randint(0, x_train_hr.shape[0], size=batch_size)

                image_batch_hr = x_train_hr[rand_nums]
                image_batch_lr = x_train_lr[rand_nums]

                generated_images_sr = generator.predict(image_batch_lr)

                real_data_Y = np.ones(batch_size) - np.random.random_sample(batch_size) * 0.2
                fake_data_Y = np.random.random_sample(batch_size) * 0.2

                discriminator.trainable = True
                d_loss_real = discriminator.train_on_batch(image_batch_hr, real_data_Y)

                d_loss_fake = discriminator.train_on_batch(generated_images_sr, fake_data_Y)
                discriminator_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

                rand_nums = np.random.randint(0, x_train_hr.shape[0], size=batch_size)
                image_batch_hr = x_train_hr[rand_nums]
                image_batch_lr = x_train_lr[rand_nums]
                gan_Y = np.ones(batch_size) - np.random.random_sample(batch_size) * 0.2
                discriminator.trainable = False
                gan_loss = gan.train_on_batch(image_batch_lr, [image_batch_hr, gan_Y])
                image_batch_hr = Utils.denormalize(x_test_hr)
                image_batch_lr = x_test_lr
                gen_img = generator.predict(image_batch_lr)
                generated_image = Utils.denormalize(gen_img)

                hr_img = [image_batch_hr[0]]
                lr_img = [image_batch_lr[0]]

                sr_img = np.reshape(generated_image[0], (-1, image_height, image_width, 3))

                summary = sess.run(tf_summary.merged,
                                   feed_dict={tf_summary.sr_img: sr_img, tf_summary.lr_img: lr_img,
                                              tf_summary.hr_img: hr_img, tf_summary.d_loss: discriminator_loss,
                                              tf_summary.gan_loss: gan_loss[0]})

            writer.add_summary(summary, e)

            # print("discriminator_loss : %f" % discriminator_loss)
            # print("gan_loss :", gan_loss)
            gan_loss = str(gan_loss)

            loss_file = open(model_save_dir + 'losses.txt', 'a')
            loss_file.write('epoch%d : gan_loss = %s ; discriminator_loss = %f\n' % (e, gan_loss, discriminator_loss))
            loss_file.close()

            if e == 1 or e % 100 == 0:
                Utils.plot_generated_images(output_dir, e, generator, x_test_hr, x_test_lr)

            if e % 1 == 0:
                generator.save(model_save_dir + 'gen_model%d.h5' % e)
                discriminator.save(model_save_dir + 'dis_model%d.h5' % e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_dir', action='store', dest='input_dir', default=image_folder,
                        help='Path for input images')

    parser.add_argument('-o', '--output_dir', action='store', dest='output_dir', default='./output/',
                        help='Path for Output images')

    parser.add_argument('-m', '--model_save_dir', action='store', dest='model_save_dir', default='./model/',
                        help='Path for model')

    parser.add_argument('-b', '--batch_size', action='store', dest='batch_size', default=batch_size,
                        help='Batch Size', type=int)

    parser.add_argument('-e', '--epochs', action='store', dest='epochs', default=amount,
                        help='number of iterations for training', type=int)

    parser.add_argument('-n', '--number_of_images', action='store', dest='number_of_images', default=1032,
                        help='Number of Images', type=int)

    parser.add_argument('-r', '--train_test_ratio', action='store', dest='train_test_ratio', default=0.8,
                        help='Ratio of train and test Images', type=float)

    values = parser.parse_args()
    train(
        values.epochs, values.batch_size, values.input_dir, values.output_dir,
        values.model_save_dir, values.number_of_images, values.train_test_ratio)













