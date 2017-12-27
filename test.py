import os,random
os.environ['KERAS_BACKEND'] = 'tensorflow'
import numpy as np
from scipy.stats import norm
from skimage.io import imsave
from keras.layers import Input, Dense, Lambda, Flatten, Reshape, merge
from keras.layers import Convolution2D, Embedding,Deconvolution2D, Activation
from keras.layers.convolutional import Conv2DTranspose,Conv2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.optimizers import RMSprop, Adam
from keras import backend as K
from keras import objectives
# import pickle
from skimage.data import imread
from skimage.transform import resize

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
nb_epoch = 250
nb_train = 160000
nb_test = 5000

save_frq = 1
train_dis_frq = 1

# samples_name = ['000.png','005.png','009.png','018.png','028.png','038.png','054.png','057.png','063.png',]
samples_name = ['chris.png','trump.png']

def vertify(img):
    print(img.shape)
    print(np.max(img),np.min(img))

# ------------------------build vae------------------- #
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

dec = Dense(intermediate_dim,name='dec_hid')(z)
dec = Dense(8*8*256,activation='relu',name='fc_latent_in')(dec)
dec = BatchNormalization(name='debatch_1')(dec)
dec = Activation('relu')(dec)
dec = Reshape((8,8,256),name='reshape')(dec)
dec = Deconvolution2D(256,(5,5),strides=(2,2),padding='same',name='deconv_1')(dec)
dec = BatchNormalization(name='debatch_2')(dec)
dec = Activation('relu')(dec)
dec = Deconvolution2D(128,(5,5),strides=(2,2),padding='same',activation='relu',name='deconv_2')(dec)
dec = BatchNormalization(name='debatch_3')(dec)
dec = Activation('relu')(dec)
dec = Deconvolution2D(32,(5,5),strides=(2,2),padding='same',activation='relu',name='deconv_3')(dec)
dec = BatchNormalization(name='debatch_4')(dec)
dec = Activation('relu')(dec)
dec_out = Conv2D(img_chns,(5,5),padding='same',activation='tanh',name='dec_conv_4')(dec)

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
vae.load_weights('./vae_result/params_vae_epoch_050.hdf5')

# ------------------bulid vae end-------------------#

# ------------------build vaed----------------------#
x = Input(shape=(img_rows,img_cols,img_chns))
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

z = Lambda(sampling, output_shape=(latent_dim,),name='sample_layer')([z_mean, z_log_var])

dec = Dense(intermediate_dim,name='dec_hid')(z)
dec = Dense(8*8*256,activation='relu',name='fc_latent_in')(dec)
dec = BatchNormalization(name='debatch_1')(dec)
dec = Activation('relu')(dec)
dec = Reshape((8,8,256),name='reshape')(dec)
dec = Deconvolution2D(256,(5,5),strides=(2,2),padding='same',name='deconv_1')(dec)
dec = BatchNormalization(name='debatch_2')(dec)
dec = Activation('relu')(dec)
dec = Deconvolution2D(128,(5,5),strides=(2,2),padding='same',name='deconv_2')(dec)
dec = BatchNormalization(name='debatch_3')(dec)
dec = Activation('relu')(dec)
dec = Deconvolution2D(32,(5,5),strides=(2,2),padding='same',name='deconv_3')(dec)
dec = BatchNormalization(name='debatch_4')(dec)
dec = Activation('relu')(dec)
dec_out = Conv2D(img_chns,(5,5),padding='same',activation='tanh',name='dec_conv_4')(dec)

vaed  = Model(x,dec_out,name='vae')
vaed.compile(optimizer=RMSprop(lr=0.0003),loss=vae_loss)
vaed.load_weights('./vaed_result/params_vaed_epoch_050.hdf5')

# ---------------build vaed end-------------------- #
samples = list()
for name in samples_name:
    img = imread(name)
    if img.shape != (64,64,3):
        img = resize(img,(64,64,3))
    if np.max(img) > 2:
        img = img / 255.0
    samples.append(img)

for sample,name in zip(samples,samples_name):
    # vertify(sample)
    # assert False
    ori = sample
    if np.max(ori) < 2:
        ori = ori * 255
    recon_name = name.split('.')[0] + '_recon.' + name.split('.')[1]
    aegan_y = imread(os.path.join('/home/media/github/autoencoding_beyond_pixels/out/celeba_reconganweight1.0e-06_recondepth9_nodisaerecon/reconstructions',recon_name))
    name = name.split('.')[0] + '_comp.' + name.split('.')[1]
    x = np.array([sample]*100)
    vae_y = vae.predict(x,batch_size=batch_size)[0,:,:,:] * 255
    vaed_y = vaed.predict(x,batch_size=batch_size)[0,:,:,:] * 255
    y = np.concatenate((ori,vaed_y,vae_y,aegan_y),axis=1).astype('uint8')
    imsave(name,y)
    # assert False
