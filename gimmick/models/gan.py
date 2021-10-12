import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint

class Generator():
    def __init__(self, optimizer, learning_rate, loss_function, metrics):
        super(Generator, self).__init__()

        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.metrics = metrics

    def build_graph(self, z_dim, im_dim, hidden_dim):
        '''
        Generator Class
        Parameters:
            z_dim: the dimension of the noise vector, a scalar
            im_dim: the dimension of the images, fitted for the dataset used, a scalar
              (MNIST images are 28 x 28 = 784 so that is your default)
            hidden_dim: the inner dimension, a scalar
        '''
        model = keras.Sequential(name="generator")
        model.add(layers.InputLayer(input_shape=z_dim))
        for i in range(1, 4):
            model.add(layers.Dense(hidden_dim * i, name="generator_layer_" + str(i)))
            model.add(layers.BatchNormalization())
            model.add(layers.Activation('relu'))
        model.add(layers.Dense(im_dim, name="generator_layer_last" + str(i)))
        model.add(layers.Activation('sigmoid'))

        self.optimizer.learning_rate = self.learning_rate
        model.compile(optimizer=self.optimizer, loss=self.loss_function, metrics=self.metrics)
        print(model.summary())
        return model

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
        im_dim = np.prod(images_shape)

        generatorobj = Generator(self.optimizer, self.learning_rate, self.loss_function, self.metrics)
        self.generator = generatorobj.build_graph(z_dim=self.code_length, im_dim=im_dim, hidden_dim=128)
        print("generator:", self.generator)


    # ''' This function train model '''
    # def train(self, images_train, images_test, epochs=10, batch_size=16, validation_split=0.2):
    #
    #     startime = datetime.now()
    #
    #     checkpoint = ModelCheckpoint(constants.DEFAULT_TF_MODELFILE, verbose=0, monitor='val_loss', save_best_only=True, mode='auto')
    #     early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    #
    #     print("================================= Training ===================================")
    #     model = self.model
    #     model.fit(images_train, images_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split,
    #               callbacks=[checkpoint, early_stopping], shuffle=True)
    #
    #     model.save(constants.DEFAULT_TF_MODELFILE) # Save Best model to disk
    #     print("Total Training time:", datetime.now() - startime)
    #
    #     print("================================= Evaluating ===================================")
    #     model.evaluate(images_test, images_test, batch_size=batch_size, verbose=True)
