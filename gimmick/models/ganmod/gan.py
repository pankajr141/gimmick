import math
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from tqdm import tqdm
from datetime import datetime

class Generator():
    def __init__(self, optimizer, learning_rate, loss_function, metrics, num_layers=None):
        super(Generator, self).__init__()

        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.metrics = metrics
        self.num_layers = num_layers

    def build_graph(self, images_shape, code_length, hidden_dim):
        '''
        Generator Class
        Parameters:
            code_length: the dimension of the noise vector, a scalar
            total_image_pixels: the dimension of the images, fitted for the dataset used, a scalar
              (MNIST images are 28 x 28 = 784 so that is your default)
            hidden_dim: the inner dimension, a scalar
        '''
        print("======================================= Generator =========================================")
        total_image_pixels = np.prod(images_shape)
        model = keras.Sequential(name="generator")
        model.add(layers.InputLayer(input_shape=code_length))

        num_layers = self.num_layers if self.num_layers else max( 3, int(total_image_pixels / hidden_dim) - 1)
        print('layers', num_layers)

        for i in range(num_layers):
            model.add(layers.Dense(hidden_dim * i, name="generator_layer_" + str(i)))
            model.add(layers.BatchNormalization())
            model.add(layers.Activation('relu'))
        model.add(layers.Dense(total_image_pixels, name="generator_layer_last" + str(i)))
        model.add(layers.Activation('sigmoid'))
        model.add(layers.Reshape(images_shape))
        self.optimizer.learning_rate = self.learning_rate
        #model.compile(optimizer=self.generator_optimizer, loss=self.loss_function, metrics=self.metrics)
        print(model.summary())
        self.model = model

class Discriminator():
    def __init__(self, optimizer, learning_rate, loss_function, metrics, num_layers=None):
        super(Discriminator, self).__init__()

        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.metrics = metrics
        self.num_layers = num_layers

    def build_graph(self, images_shape, hidden_dim):
        '''
        Discriminator Class
        Parameters:
            code_length: the dimension of the noise vector, a scalar
            total_image_pixels: the dimension of the images, fitted for the dataset used, a scalar
              (MNIST images are 28 x 28 = 784 so that is your default)
            hidden_dim: the inner dimension, a scalar
        '''
        print("======================================= Discriminator =========================================")
        total_image_pixels = np.prod(images_shape)
        num_layers = self.num_layers if self.num_layers else int(math.log(total_image_pixels, 2)) - int(math.log(hidden_dim, 2))
        print('layers', num_layers)

        model = keras.Sequential(name="discriminator")
        model.add(layers.InputLayer(input_shape=images_shape))

        model.add(layers.Flatten())
        for i in range(num_layers, 0, -1):
            model.add(layers.Dense(hidden_dim * i, name="discriminator_layer_" + str(i)))
            model.add(tf.keras.layers.LeakyReLU())

        model.add(layers.Dense(1))
        #model.add(layers.Activation('sigmoid'))

        self.optimizer.learning_rate = self.learning_rate
        #model.compile(optimizer=self.discriminator_optimizer, loss=self.loss_function, metrics=self.metrics)
        print(model.summary())
        self.model = model


class Model():

    def __init__(self, learning_rate=None, optimizer=None, optimizer_keys=None, loss_function=None, loss_function_keys=None, metrics=None, metrics_keys=None,
                 code_length=8, num_generator_layers=-1, num_discriminator_layers=-1):

        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.metrics = metrics
        self.code_length = code_length

        self.loss_function_keys = loss_function_keys
        self.optimizer_keys = optimizer_keys
        self.metrics_keys = metrics_keys

    def build_model_graph(self, images_shape):
        print("images_shape:", images_shape)
        total_image_pixels = np.prod(images_shape)
        self.generator = Generator(self.optimizer, self.learning_rate, self.loss_function, self.metrics)
        self.generator.build_graph(images_shape, self.code_length, hidden_dim=128)

        self.discriminator = Discriminator(self.optimizer, self.learning_rate, self.loss_function, self.metrics)
        self.discriminator.build_graph(images_shape, hidden_dim=128)

        print("generator:", self.generator)
        print("discriminator:", self.discriminator)

    @tf.function
    def train_step(self, images, code_length, batch_size):

        def discriminator_loss(pred_real, pred_fake):
            real_loss = self.loss_function(tf.ones_like(pred_real), pred_real) # compares discriminator's predictions on real images to an array of 1s,
            fake_loss = self.loss_function(tf.zeros_like(pred_fake), pred_fake) # compares discriminator's predictions on fake (generated) images to an array of 0s.
            total_loss = real_loss + fake_loss
            return total_loss

        def generator_loss(pred_fake):
            # Loss will be low is fake images have a higher score by discriminator of being 1, else viseversa
            return self.loss_function(tf.ones_like(pred_fake), pred_fake) # compares discriminators decisions on the generated images to an array of 1s.

        noise = tf.random.normal([batch_size, code_length])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator.model(noise, training=True)

            pred_real = self.discriminator.model(images, training=True)
            pred_fake = self.discriminator.model(generated_images, training=True)

            gen_loss = generator_loss(pred_fake)
            disc_loss = discriminator_loss(pred_real, pred_fake)

        # print("Gen loss:", tf.print(gen_loss), "Disc loss:", disc_loss)
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.model.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.model.trainable_variables)

        self.generator.optimizer.apply_gradients(zip(gradients_of_generator, self.generator.model.trainable_variables))
        self.discriminator.optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.model.trainable_variables))
        #print(type(gen_loss))
        #return gen_loss.numpy(), disc_loss.numpy()
        return gen_loss, disc_loss

    def train(self, images_train, images_test, epochs=10, batch_size=16, validation_split=0.2):

        # checkpoint1 = ModelCheckpoint(constants.DEFAULT_TF_GENERATOR_MODELFILE, verbose=0, monitor='val_loss', save_best_only=True, mode='auto')
        # checkpoint2 = ModelCheckpoint(constants.DEFAULT_TF_DISCRIMINATOR_MODELFILE, verbose=0, monitor='val_loss', save_best_only=True, mode='auto')
        #
        checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator.optimizer, discriminator_optimizer=self.discriminator.optimizer,
                                         generator=self.generator.model,  discriminator=self.discriminator.model)

        # early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        train_dataset = tf.data.Dataset.from_tensor_slices(images_train).shuffle(batch_size * 4).batch(batch_size)

        print("================================= Training ===================================")

        generator = self.generator
        discriminator = self.discriminator
        code_length = 8

        startime = datetime.now()
        for i in range(epochs):
            total_generator_loss = 0
            total_discriminator_loss = 0
            steps_t = images_train.shape[0] / batch_size

            batch_log = tqdm(total=steps_t, desc="f'epoch: {i}, generator_loss(t): {mean_generator_loss:10.8f}, discriminator_loss(t): {mean_discriminator_loss: 10.8f}'", position=0)

            for j, image_batch in enumerate(train_dataset):
                generator_loss, discriminator_loss = self.train_step(image_batch, code_length, batch_size)

                total_discriminator_loss += K.eval(discriminator_loss)
                total_generator_loss += K.eval(generator_loss)

                mean_generator_loss = total_generator_loss / j
                mean_discriminator_loss = total_discriminator_loss / j

                batch_log.update(1)
                batch_log.set_description(f'epoch: {i}, generator_loss(t): {mean_generator_loss:10.8f}, discriminator_loss(t): {mean_discriminator_loss: 10.8f}')

            # generator.fit(images_train, images_train, batch_size=batch_size, epochs=1, validation_split=validation_split,
            #       callbacks=[checkpoint, early_stopping], shuffle=True)
        #model.save(constants.DEFAULT_TF_MODELFILE) # Save Best model to disk
        print("Total Training time:", datetime.now() - startime)

        print("================================= Evaluating ===================================")
        model.evaluate(images_test, images_test, batch_size=batch_size, verbose=True)
