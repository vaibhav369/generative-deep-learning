from keras.datasets import mnist
from keras.layers import Dense, Conv2D, Conv2DTranspose, Flatten, Reshape, BatchNormalization
from keras.layers import LeakyReLU, Dropout, Lambda, Input, Activation
from keras.optimizers import Adam
from keras.models import Model, Sequential
import keras.backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping

import numpy as np
import os


class VariationalAutoencoder:

    def __init__(self):
        self.vae_input_shape = (28, 28, 1)
        self.encoder_num_filters = [32, 64, 64, 64]
        self.encoder_kernel_sizes = [3, 3, 3, 3]
        self.encoder_strides = [1, 2, 2, 1]

        self.decoder_num_filters = [64, 64, 32, 1]
        self.decoder_kernel_sizes = [3, 3, 3, 3]
        self.decoder_strides = [1, 2, 2, 1]

        self.batch_normalization = True
        self.dropout = True
        self.latent_dim = 2

        self._build()

    def _build(self):

        # Encoder
        encoder_input = Input(shape=self.vae_input_shape, name='encoder_input')
        x = encoder_input

        for i, (filter_num, kernel_size, stride) in enumerate( zip(self.encoder_num_filters, self.encoder_kernel_sizes, self.encoder_strides) ):
            x = Conv2D(filters=filter_num,
                       kernel_size=kernel_size,
                       strides=stride,
                       padding='same',
                       name='encoder_conv_' + str(i+1))(x)
            if self.batch_normalization:
                x = BatchNormalization()(x)
            x = LeakyReLU()(x)
            if self.dropout:
                x = Dropout(rate=0.25)(x)

        shape_before_flatten = K.int_shape(x)[1:]
        x = Flatten()(x)

        self.mu = Dense(self.latent_dim, name='mu')(x)
        self.logvar = Dense(self.latent_dim, name='logvar')(x)

        self.encoder_mu_logvar = Model(encoder_input, (self.mu, self.logvar))

        def sampling(args):
            mu, logvar = args
            epsilon = K.random_normal(shape=K.shape(mu), mean=0., stddev=1.)
            return mu + K.exp(logvar/2) * epsilon

        encoder_output = Lambda(sampling, name=('encoder_output'))([self.mu, self.logvar])
        self.encoder = Model(encoder_input, encoder_output)

        # Decoder
        decoder_input = Input(shape=(self.latent_dim, ), name='decoder_input')
        x = decoder_input

        x = Dense(np.prod(shape_before_flatten))(x)
        x = Reshape(shape_before_flatten)(x)

        for i, (filter_num, kernel_size, stride) in enumerate( zip(self.decoder_num_filters, self.decoder_kernel_sizes, self.decoder_strides) ):
            x = Conv2DTranspose(filters=filter_num,
                                kernel_size=kernel_size,
                                strides=stride,
                                padding='same',
                                name='decoder_conv_transposed_' + str(i+1))(x)
            if self.batch_normalization:
                x = BatchNormalization()(x)
            if i < len(self.decoder_kernel_sizes) - 1:
                x = LeakyReLU()(x)
            else:
                x = Activation('sigmoid')(x)
            if self.dropout:
                x = Dropout(rate=0.25)(x)

        decoder_output = x
        self.decoder = Model(decoder_input, decoder_output)

        # VAE
        vae_input = encoder_input
        vae_output = self.decoder( self.encoder(encoder_input) )
        self.vae = Model(vae_input, vae_output)

    def compile(self, learning_rate, r_loss_factor):
        self.learning_rate = learning_rate

        def vae_r_loss(y_true, y_pred):
            r_loss = K.mean( K.square(y_true - y_pred), axis=[1, 2, 3] )
            return r_loss_factor * r_loss

        def vae_kl_loss(y_true, y_pred):
            kl_loss = -0.5 * K.sum(1 + self.logvar - K.square(self.mu) - K.exp(self.logvar), axis=1)
            return kl_loss

        def vae_loss(y_true, y_pred):
            r_loss = vae_r_loss(y_true, y_pred)
            kl_loss = vae_kl_loss(y_true, y_pred)
            return r_loss + kl_loss

        optimizer = Adam(lr=learning_rate)
        self.vae.compile(optimizer=optimizer, loss=vae_loss, metrics=[vae_r_loss, vae_kl_loss])

    def save(self, filename='vae.hdf5'):
        models_dir = 'models'
        if not os.path.exists(models_dir):
            os.mkdir(models_dir)
        self.model_filepath = models_dir + filename
        self.vae.save_weights(models_dir + filename)


    def train(self, x_train, batch_size, epochs):
        self.vae.fit(x_train,
                     x_train,
                     batch_size=batch_size,
                     shuffle=True,
                     epochs=epochs)


if __name__ == '__main__':

    autoencoder = VariationalAutoencoder()
