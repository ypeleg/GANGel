
# coding: utf-8

import matplotlib 
matplotlib.use('Agg')
import cv2
import sys
import os
import shutil
sys.path.insert(0,'../..')

from AutoGAN import GAN
from AutoGAN.schemes.CycleGAN_TrainingScheme import CycleGAN_TrainingScheme

import keras

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import Conv1D, Conv2D, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.layers import UpSampling2D, LeakyReLU, Lambda, Add, Multiply, Activation, Conv2DTranspose
from keras.layers import Cropping2D, ZeroPadding2D, Flatten, Subtract, Input, add, multiply
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam 

import matplotlib.pyplot as plt
import numpy as np
import scipy
from skimage.transform import resize
import glob
from random import shuffle

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.95
config.gpu_options.visible_device_list = "1"
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

def build_generator(gf, size):
    """U-Net Generator"""

    def conv2d(layer_input, filters, f_size=4):
        """Layers used during downsampling"""
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        d = BatchNormalization()(d)
        return d

    def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
        """Layers used during upsampling"""
        u = UpSampling2D(size=2)(layer_input)
        u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
        if dropout_rate:
            u = Dropout(dropout_rate)(u)
        u = BatchNormalization()(u)
        u = Concatenate()([u, skip_input])
        return u

    # Image input
    d0 = Input(shape=size)

    # Downsampling
    d1 = conv2d(d0, gf)
    d2 = conv2d(d1, gf*2)
    d3 = conv2d(d2, gf*4)
    d4 = conv2d(d3, gf*8)
    d5 = conv2d(d4, gf*16)

    # Upsampling
    u0 = deconv2d(d5, d4, gf*16)
    u1 = deconv2d(u0, d3, gf*4)
    u2 = deconv2d(u1, d2, gf*2)
    u3 = deconv2d(u2, d1, gf)

    u4 = UpSampling2D(size=2)(u3)
    output_img = Conv2D(3, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)

    return Model(d0, output_img)

def build_discriminator(df, size):

    def d_layer(layer_input, filters, f_size=4, normalization=True):
        """Discriminator layer"""
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if normalization:
            d = BatchNormalization()(d)
        return d

    img = Input(shape=size)

    d1 = d_layer(img, df, normalization=False)
    d2 = d_layer(d1, df*2)
    d3 = d_layer(d2, df*4)
    d4 = d_layer(d3, df*8)
    d5 = d_layer(d4, df*16)

    validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d5)

    return Model(img, validity)


def load_data(dataset):
    filelist_A = glob.glob('./%s/trainA/*.jpg' % dataset)
    shuffle(filelist_A)
    A = np.array([cv2.resize(plt.imread(file), (256, 256)) for file in filelist_A])
    print(A.shape)
    A = (2. * A/255.) - 1.
    
    filelist_B = glob.glob('./%s/trainB/*.jpg' % dataset)
    shuffle(filelist_B)
    B = np.array([cv2.resize(plt.imread(file), (256, 256)) for file in filelist_B if plt.imread(file).shape[-1] == 3 ])
    print(B.shape)
    B = (2. * B/255.) - 1.
    return A, B

def load_data_test(dataset):
    filelist_A = glob.glob('./%s/testA/*.jpg' % dataset)
    shuffle(filelist_A)
    A = np.array([plt.imread(file) for file in filelist_A])
    print(A.shape)
    A = (2. * A/255.) - 1.
    
    filelist_B = glob.glob('./%s/testB/*.jpg' % dataset)
    shuffle(filelist_B)
    B = np.array([plt.imread(file) for file in filelist_B])
    print(B.shape)
    B = (2. * B/255.) - 1.
    return A, B



class save_images(keras.callbacks.Callback):
    def __init__(self, model, A, B, freq, dataset):
        super(save_images, self).__init__()
        try:
            import os
            os.makedirs('images/%s' % dataset)
        except:
            pass
        self.full_model = model
        self.A = A
        self.B = B
        self.epoch = 0
        self.freq = freq
        self.dataset = dataset
    def sample_images(self, epoch, batch_i, A=None, B=None):
        r, c = 2, 3

        # Demo (for GIF)
        if A is None:
            if 'apple2orange' in self.dataset:
                A = np.array([cv2.resize( plt.imread('./apple2orange/trainA/7dayjamberrychallenge1.jpg','jpg'), (256, 256)) ])
            else:
                A = np.array([cv2.resize( plt.imread('./horse2zebra/trainA/7dayjamberrychallenge1.jpg','jpg'), (256, 256))  ])
            A = (2. * A/255.) - 1.
        if B is None:
            if 'apple2orange' in self.dataset:
                B = np.array([cv2.resize(plt.imread('./apple2orange/trainB/99dd70131383c44cd23e766a1cc70724.jpg','jpg'), (256, 256)) ])
            else:
                B = np.array([cv2.resize(plt.imread('./horse2zebra/trainB/99dd70131383c44cd23e766a1cc70724.jpg','jpg'), (256,256) )  ])
            B = (2. * B/255.) - 1.

        # Translate images to the other domain
        fake_A = self.full_model.generator_model()[0].predict(B)
        fake_B = self.full_model.generator_model()[1].predict(A)
        # Translate back to original domain
        reconstr_A = self.full_model.generator_model()[0].predict(fake_B)
        reconstr_B = self.full_model.generator_model()[1].predict(fake_A)

        gen_imgs = np.concatenate([A, fake_B, reconstr_A, B, fake_A, reconstr_B])

        # Rescale images 0 - 1
        #gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Original', 'Translated', 'Reconstructed']
        fig, axs = plt.subplots(r, c,figsize=(10,10))
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow((gen_imgs[cnt]+1.)/2.)
                axs[i, j].set_title(titles[j])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%s/%d_%d.png" % (self.dataset, epoch, batch_i))
        #plt.show()
        plt.close()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        #print('started epoch %d' % epoch)
    def on_epoch_end(self, epoch, logs=None):
        pass #print(logs)
    def on_batch_end(self, batch, logs=None):
        if batch % self.freq == 0:
            self.sample_images(self.epoch, batch)
            #print('sampled data at epoch %d , batch %d' % (self.epoch, batch))
    def on_train_end(self, logs=None):
        for i in range(1, self.A.shape[0]):
            try:
                self.sample_images(self.epoch+1, i-1, self.A[i-1:i], self.B[i-1:i])
            except:
                continue
    
class pretrain_model(keras.callbacks.Callback):
    def __init__(self, my_model, x, y, epochs, batch_size, loss, metrics, optimizer):        
        self.my_model = my_model
        self.x, self.y = x, y
        self.batch_size = batch_size
        self.epochs = epochs
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer
    def on_train_begin(self, logs=None):
        self.my_model.compile(loss=self.loss, metrics=self.metrics, optimizer=self.optimizer)
        self.my_model.fit(self.x, self.y, epochs=self.epochs, batch_size=self.batch_size, verbose=1, shuffle=True, validation_split=0.2)



# In[ ]:

A, B = load_data('apple2orange')
A_test, B_test = load_data_test('apple2orange')

def cyclegan(image_A, image_B):
    model = GAN(generator=[build_generator(32, image_A.shape), build_generator(32, image_B.shape)], 
                discriminator=[build_discriminator(32, image_A.shape),build_discriminator(32, image_B.shape)])
    optimizer = keras.optimizers.Adam(0.0002, 0.5)
    optimizerD = keras.optimizers.Adam(0.0001, 0.5)
    try:
        shutil.rmtree('./images/apple2orange_cyclegan')
    except:
        pass
    discriminator_kwargs = {'loss':'mse', 'optimizer': optimizerD}
    generator_kwargs = {'optimizer': optimizer,
                        'translation_weight':1, 'cycle_weight':10, 'identity_weight':1,
                        'translation_loss':'mse', 'cycle_loss':'mae', 'identity_loss':'mae'}
    model.compile(training_scheme=CycleGAN_TrainingScheme(),
                  generator_kwargs=generator_kwargs, discriminator_kwargs=discriminator_kwargs)
    return model

print A[0].shape
print B[0].shape
model = cyclegan(A[0], B[0])
 
#model.summary(True)
#get_ipython().magic(u'matplotlib inline')
model.fit(x=A, y=B, epochs=5, steps_per_epoch=1000, batch_size=1,
          generator_callbacks=[save_images(model, A_test, B_test, 100,'apple2orange_cyclegan')], verbose=1)


# In[ ]:



