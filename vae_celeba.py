
'''This script demonstrates how to build a variational autoencoder
with Keras and deconvolution layers.

Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
'''

'''
trained on celeba
'''

import os,random
os.environ['KERAS_BACKEND'] = 'tensorflow'

from skimage.io import imsave

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda, Flatten, Reshape, merge
from keras.layers import Convolution2D, Embedding,Deconvolution2D, Activation
from keras.layers.convolutional import Conv2DTranspose,Conv2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.optimizers import RMSprop
from keras import backend as K
from keras import objectives
import pickle

import load_dataset

# input image dimensions
img_rows, img_cols, img_chns = 64, 64, 3
is_gray = False
# number of convolutional filters to use

batch_size = 100
latent_dim = 128
intermediate_dim = 2048
epsilon_std = 1.0
nb_epoch = 50
nb_train = 160000
nb_test = 5000

x = Input(batch_shape=(batch_size,img_rows,img_cols,img_chns),name='img_input')
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

z_mean = Dense(latent_dim,name='fc_zmean')(enc_out)
z_log_var = Dense(latent_dim,name='fc_zlogvar')(enc_out)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim),
                              mean=0.,stddev=epsilon_std)
    return z_mean + K.exp(z_log_var) * epsilon
z = Lambda(sampling, output_shape=(latent_dim,),name='sample_layer')([z_mean, z_log_var])

dec_hid = Dense(intermediate_dim,name='dec_hid')
fc = Dense(8*8*256,activation='relu',name='fc_latent_in')
debatch_1 = BatchNormalization(name='debatch_1')
relu_1 = Activation('relu')
reshape = Reshape((8,8,256),name='reshape')
deconv_1 = Deconvolution2D(256,(5,5),strides=(2,2),padding='same',name='deconv_1')
debatch_2 = BatchNormalization(name='debatch_2')
relu_2 = Activation('relu')
deconv_2 = Deconvolution2D(128,(5,5),strides=(2,2),padding='same',activation='relu',name='deconv_2')
debatch_3 = BatchNormalization(name='debatch_3')
relu_3 = Activation('relu')
deconv_3 = Deconvolution2D(32,(5,5),strides=(2,2),padding='same',activation='relu',name='deconv_3')
debatch_4 = BatchNormalization(name='debatch_4')
relu_4 = Activation('relu')
conv_4 = Conv2D(img_chns,(5,5),padding='same',activation='tanh',name='dec_conv_4')

# build decoder
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


def vae_loss(x, x_decoded_mean):
    # NOTE: binary_crossentropy expects a batch_size by dim
    # for x and x_decoded_mean, so we MUST flatten these!
    x = K.flatten(x)
    x_decoded_mean = K.flatten(x_decoded_mean)
    xent_loss = img_rows * img_cols * objectives.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return xent_loss + kl_loss

vae = Model(x,dec_out,name='vae')
vae.compile(optimizer=RMSprop(lr=0.0003),loss=vae_loss)

data_raw = load_dataset.load_celeba(nb_train+nb_test,False)

x_train = data_raw[0:nb_train,:,:,:]
x_test = data_raw[nb_train:nb_train+nb_test,:,:,:]
print('x_train.shape:', x_train.shape)
print('x_test,shape:', x_test.shape)
print(np.max(x_test),np.min(x_test))

vae.fit(x_train, x_train,
        shuffle=True,
        epochs=nb_epoch,
        batch_size=batch_size,
        validation_data=(x_test, x_test))



# build a model to project inputs on the latent space
encoder = Model(x, z_mean,name='encoder')

# build a digit generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,),name='dec_img_input')

_dec = dec_hid(decoder_input)
_dec = fc(_dec)
_dec = debatch_1(_dec)
_dec = relu_1(_dec)
_dec = reshape(_dec)
_dec = deconv_1(_dec)
_dec = debatch_2(_dec)
_dec = relu_2(_dec)
_dec = deconv_2(_dec)
_dec = debatch_3(_dec)
_dec = relu_3(_dec)
_dec = deconv_3(_dec)
_dec = debatch_4(_dec)
_dex = relu_4(_dec)
_dec_out = conv_4(_dec)

generator = Model(decoder_input, _dec_out)


indexs = np.random.randint(0,x_test.shape[0],100)

real_images = x_test[indexs,:,:,:]
gen_images = vae.predict(real_images,batch_size=batch_size)


vae.save_weights(
'vae_result/params_vae_epoch_{0:03d}.hdf5'.format(nb_epoch), True)
generator.save_weights(
'vae_result/params_generator_epoch_{0:03d}.hdf5'.format(nb_epoch), True)
encoder.save_weights(
'vae_result/params_encoder_epoch_{0:03d}.hdf5'.format(nb_epoch), True)

n = 10
show_real = True
index = 0

figure = np.zeros((img_rows * n, img_cols * n, img_chns))
for i in range(n):
    for j in range(n):
        if show_real:
            figure[i*img_rows:(i+1)*img_cols,j*img_cols:(j+1)*img_cols,:] = real_images[index,:,:,:]
        else:
            figure[i*img_rows:(i+1)*img_cols,j*img_cols:(j+1)*img_cols,:] = gen_images[index,:,:,:]
            index += 1
        show_real = not show_real

imsave('vae_result/plot_epoch_{0:03d}_generated.png'.format(nb_epoch),figure)

# for i in range(len(show_images)):
#   plt.subplot(8,8,i+1)
#   plt.imshow(show_images[i,:,:,:])
#   plt.axis('off')
#
# plt.show()
