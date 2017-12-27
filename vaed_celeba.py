
'''This script demonstrates how to build a variational autoencoder
with Keras and deconvolution layers.

Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
'''

'''
trained on celeba
'''

import os,random
os.environ['KERAS_BACKEND'] = 'tensorflow'

# from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from skimage.io import imsave  #io : Reading, saving, and displaying images and video

from keras.layers import Input, Dense, Lambda, Flatten, Reshape, merge
from keras.layers import Convolution2D, Embedding, Deconvolution2D, Activation
from keras.layers.convolutional import Conv2DTranspose, Conv2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.optimizers import RMSprop, Adam
from keras import backend as K
from keras import objectives
from keras.utils.generic_utils import Progbar
from keras.utils.vis_utils import plot_model

import pickle
import load_dataset

# input image dimensions
img_rows, img_cols, img_chns = 64, 64, 3

if img_chns != 1:
    is_gray =  False
else:
    is_gray = True


batch_size = 100
latent_dim = 128
intermediate_dim = 2048
epsilon_std = 1.0
nb_epoch = 50
nb_train = 160000
nb_test = 5000

save_frq = 1
train_dis_frq = 1

def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val

# build vae
x = Input(shape=(img_rows, img_cols, img_chns))
enc = Conv2D(64, (5, 5), padding="same", strides=(2, 2), name="conv1")(x)
enc = BatchNormalization(name='batch1')(enc)
enc = Activation('relu')(enc)
enc = Conv2D(128, (5, 5), padding="same", strides=(2, 2), name="conv2")(enc)
enc = BatchNormalization(name='batch2')(enc)
enc = Activation('relu')(enc)
enc = Conv2D(256, (5, 5), padding="same", strides=(2, 2), name="conv3")(enc)
enc = BatchNormalization(name='batch3')(enc)
enc = Activation('relu')(enc)

enc = Flatten(name='flat')(enc)
enc = Dense(intermediate_dim,name='fc_enc_out')(enc)
enc = BatchNormalization(name='batch4')(enc)
enc_out = Activation('relu')(enc)

z_mean = Dense(latent_dim, name='fc_zmean')(enc_out)
z_log_var = Dense(latent_dim, name='fc_zlogvar')(enc_out)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal((batch_size, latent_dim),
    mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_var) * epsilon

z = Lambda(sampling, output_shape=(latent_dim, ),
name='sample_layer')([z_mean, z_log_var])

dec_hid = Dense(intermediate_dim,name='dec_hid')
fc = Dense(8*8*256, activation='relu', name='fc_latent_in')
debatch_1 = BatchNormalization(name='debatch_1')
relu_1 = Activation('relu')
reshape = Reshape((8, 8, 256), name='reshape')
deconv_1 = Deconvolution2D(256, (5, 5), strides=(2, 2),
                           padding='same', name='deconv_1')
debatch_2 = BatchNormalization(name='debatch_2')
relu_2 = Activation('relu')
deconv_2 = Deconvolution2D(128, (5, 5), strides=(2, 2),
                           padding='same', name='deconv_2')
debatch_3 = BatchNormalization(name='debatch_3')
relu_3 = Activation('relu')
deconv_3 = Deconvolution2D(32, (5, 5),strides=(2, 2),
                           padding='same', name='deconv_3')
debatch_4 = BatchNormalization(name='debatch_4')
relu_4 = Activation('relu')
conv_4 = Conv2D(img_chns, (5, 5),padding='same',
                           activation='tanh', name='dec_conv_4')

dec = dec_hid(z)
dec = fc(dec)
dec = debatch_1(dec)
dec = relu_1(dec)
dec = reshape(dec)
dec = deconv_1(dec)
dec = debatch_2(dec)
dec = relu_2(dec)
dec = deconv_2(dec)
dec = debatch_3(dec)
dec = relu_3(dec)
dec = deconv_3(dec)
dec = debatch_4(dec)
dec = relu_4(dec)
dec_out = conv_4(dec)

vae  = Model(x, dec_out, name='vae')

# build discriminator

dis_input = Input(shape=(img_rows, img_cols, img_chns))
dis = Conv2D(32, (5, 5), padding='same', activation='relu',name='disconv_1')(dis_input)
dis = Conv2D(128, (5, 5), strides=(2, 2), padding='same', name='disconv_2')(dis)
dis = BatchNormalization(name='disbatch_1')(dis)
dis = Activation('relu')(dis)
dis = Conv2D(256, (5, 5), strides=(2, 2), padding='same', name='disconv_3')(dis)
dis = BatchNormalization(name='disbatch_2')(dis)
dis = Activation('relu')(dis)
dis = Conv2D(256, (5, 5), strides=(2, 2), padding='same', name='disconv_4')(dis)
dis = BatchNormalization(name='disbatch_3')(dis)
dis = Activation('relu')(dis)
dis = Flatten(name='disflat')(dis)
dis = Dense(512, name='disfc')(dis) # dis => dis_llike
dis = BatchNormalization(name='disbatch_4')(dis) # dis => dis_llike
dis = Activation('relu')(dis)
dis_out = Dense(1, activation='sigmoid', name='fc_out')(dis)

discriminator = Model(dis_input, dis_out,name='discriminator')
discriminator.compile(optimizer=RMSprop(lr=0.0003), loss='binary_crossentropy')

dec_out = vae(x)
make_trainable(discriminator, False)
dis_out = discriminator(dec_out)
# dis_llike = discriminator(x)
vae_gan = Model(x, [dec_out, dis_out], name='vae_gan') # dec_out => dis_like, x => [x,x]
def vae_loss(x, x_decoded_mean):
    # NOTE: binary_crossentropy expects a batch_size by dim
    # for x and x_decoded_mean, so we MUST flatten these!
    x = K.flatten(x)
    x_decoded_mean = K.flatten(x_decoded_mean)
    xent_loss = img_rows * img_cols * objectives.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return xent_loss + kl_loss
vae_gan.compile(optimizer=RMSprop(lr=0.0003), loss={'vae':vae_loss,'discriminator':'binary_crossentropy'})
vae_gan.summary()

#plot_model(vae_gan, to_file='vae_gan.png')

#load dataset
def train():
    data_raw = load_dataset.load_celeba(nb_train+nb_test, False)
    x_train = data_raw[0:nb_train, :, :, :]
    x_test = data_raw[nb_train:nb_train+nb_test, :, :, :]
    print('x_train.shape:', x_train.shape)
    print('x_test,shape:', x_test.shape)
    print(np.max(x_test), np.min(x_test))

    for epoch in range(nb_epoch):
        print('Epoch {} of {}'.format(epoch + 1, nb_epoch))

        nb_batches = int(x_train.shape[0] / batch_size)
        progress_bar = Progbar(target=nb_batches)

        epoch_vae_loss = []
        epoch_disc_loss = []

        for index in range(nb_batches):
            progress_bar.update(index)

            # get a batch of real images
            image_batch = x_train[index * (batch_size):(index + 1) * (batch_size)]

            generated_images = vae.predict(
                image_batch, verbose=0, batch_size=batch_size)


            X = np.concatenate((image_batch, generated_images))
            y = np.array([1] * (batch_size/2) + [0] * (batch_size/2))
            X_1 = np.concatenate((image_batch[0:batch_size/2], generated_images[:batch_size/2]))
            X_2 = np.concatenate((image_batch[batch_size/2:], generated_images[batch_size/2:]))

            make_trainable(discriminator, True)
            # see if the discriminator can figure itself out...
            if epoch == 0 or epoch % train_dis_frq ==0 :
                epoch_disc_loss.append(discriminator.train_on_batch(X_1, y))
                epoch_disc_loss.append(discriminator.train_on_batch(X_2, y))


            trick = np.ones(batch_size)

            epoch_vae_loss.append(vae_gan.train_on_batch(
                X_1, [X_1, trick]))
            epoch_vae_loss.append(vae_gan.train_on_batch(
                X_2, [X_2, trick]))

        if epoch % save_frq == 0:
            vae_train_loss = np.mean(np.array(epoch_vae_loss), axis=0)
            disc_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)

            print('{0:<22s} | {1:4s}'.format(
                'component', *discriminator.metrics_names))
            print('-' * 65)
            ROW_FMT = '{0:<22s} | {1:<4.2f}'
            print(ROW_FMT.format('vae', vae_train_loss[1]))
            print(ROW_FMT.format('discriminator', disc_train_loss))

            vae.save_weights('vaed_result/params_vaed_epoch_{0:03d}.hdf5'.format(epoch), True)

            samples = x_test[np.random.randint(0, x_test.shape[0],batch_size), :, :, :]
            gens = vae.predict(samples, batch_size=batch_size)

            n = 10
            show_real = True
            index = 0

            figure = np.zeros((img_rows * n, img_cols * n, img_chns))
            for i in range(n):
                for j in range(n):
                    if show_real:
                        figure[i*img_rows:(i+1)*img_cols, j*img_cols:(j+1)*img_cols, :] = samples[index, :, :, :]
                    else:
                        figure[i*img_rows:(i+1)*img_cols, j*img_cols:(j+1)*img_cols, :] = gens[index, :, :, :]
                        index += 1
                    show_real = not show_real


            imsave('vaed_result/plot_epoch_{0:03d}_generated.png'.format(epoch), figure)

def test():
    x_test = load_dataset.load_faces(40000+batch_size, False)[40000:40000+batch_size, :, :, :]
    vae.load_weights('vaed_result/params_vaed_epoch_100.hdf5')
    gens = vae.predict(x_test, batch_size=batch_size)[:10, :, :, :]

    n = 10
    show_real = True
    index = 0

    figure = np.zeros((img_rows * n, img_cols * 2, img_chns))
    for i in range(n):
        for j in range(2):
            if show_real:
                figure[i*img_rows:(i+1)*img_cols, j*img_cols:(j+1)*img_cols, :] = x_test[index, :, :, :]
            else:
                figure[i*img_rows:(i+1)*img_cols, j*img_cols:(j+1)*img_cols, :] = gens[index, :, :, :]
                index += 1
            show_real = not show_real

    imsave('ref2.png', figure)

if __name__ == '__main__':
    train()
