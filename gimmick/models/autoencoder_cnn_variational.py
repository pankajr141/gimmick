import math
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime
from gimmick import constants

class Model():
    def __init__(self, learning_rate=None, optimizer=None, optimizer_keys=None, loss_function=None, loss_function_keys=None, metrics=None, metrics_keys=None,
                 code_length=8, num_encoder_layers=-1, num_decoder_layers=-1):
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.metrics = metrics
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.code_length = code_length

        self.loss_function_keys = loss_function_keys
        self.optimizer_keys = optimizer_keys
        self.metrics_keys = metrics_keys

    ''' This function build model graph  '''
    def build_model_graph(self, images_shape):
        total_pixels = images_shape[0] * images_shape[1] * images_shape[2]

        num_encoder_layers = int(math.log(images_shape[0], 2)) - 2 #if self.num_encoder_layer == "auto" else int(self.num_encoder_layer)
        num_decoder_layers = int(math.log(images_shape[0], 2)) - 2 #if self.num_decoder_layers == "auto" else int(self.num_decoder_layers)

        log2_code = int(math.log(self.code_length, 2))
        print("num_enoder_layer:\t", num_encoder_layers)
        print("num_decoder_layers:\t", num_decoder_layers)
        print("log2_code:\t\t", log2_code)

        filter_size = (3, 3)
        pool_size = (2, 2)

        def _encoder(x):
            # x = layers.Reshape((images_shape[0], images_shape[1], images_shape[2]))(x)

            # Encoder Layer
            for i in range(1, num_encoder_layers + 1):
                neurons = 2 ** (num_encoder_layers - i + log2_code + 1) # Encoder layer size will be always greater then code_length by multiple of 2

                if i == 1 and images_shape[0] <= 16:
                    x = layers.Conv2D(neurons, filter_size, activation='relu', padding='same', strides=1, name="encoder_layer_extra1")(x)
                if i == 1 and images_shape[0] <= 8:
                    x = layers.Conv2D(neurons, filter_size, activation='relu', padding='same', strides=1, name="encoder_layer_extra2")(x)
                x = layers.Conv2D(neurons, filter_size, activation='relu', padding='same', strides=2, name="encoder_layer_" + str(i) )(x)

            # Code Layer
            x = layers.Flatten()(x)
            return x

        def _decoder(code):

            x = layers.Reshape((2, 2, int(self.code_length / 4)))(code)

            # Decoder Layer
            for i in range(1, num_decoder_layers + 1):
                neurons = 2 ** (i + log2_code)  # Decoder layer size will be always greater then code_length by multiple of 2

                if i == 1 and images_shape[0] <= 16:
                    x = layers.Conv2DTranspose(neurons, filter_size, activation='relu', padding='same', strides=1, name="decoder_layer_extra1" )(x)
                if i == 1 and images_shape[0] <= 8:
                    x = layers.Conv2DTranspose(neurons, filter_size, activation='relu', padding='same', strides=1, name="decoder_layer_extra2" )(x)
                x = layers.Conv2DTranspose(neurons, filter_size, activation='relu', padding='same', strides=2, name="decoder_layer_" + str(i) )(x)

            output = layers.Conv2DTranspose(images_shape[2], filter_size, activation='relu', padding='same', strides=2, name='decoder_layer_%s' % str(i + 1) )(x)
            return output

        input = keras.Input(shape=images_shape)

        x = _encoder(input)

        mean = layers.Dense(self.code_length, name='mean')(x)
        std = keras.activations.softplus(layers.Dense(self.code_length, name='std')(x))

        # Reparametrization trick
        epsilon = tf.random.normal(tf.stack([tf.shape(x)[0], self.code_length]), name='epsilon')
        code = tf.add(mean, epsilon * std, name='code')

        output = _decoder(code)

        optimizer =self.optimizer
        optimizer.learning_rate = self.learning_rate

        model = tf.keras.Model(inputs=input, outputs=output)
        model.compile(optimizer=optimizer, loss=self.loss_function, metrics=self.metrics)
        self.model = model

        print(model.summary())

        # Building model which generate code statistics
        model_code_generator = tf.keras.Model(inputs=input, outputs=code)
        model_code_generator.build((None, images_shape[0], images_shape[1], images_shape[2]))
        self.model_code_generator = model_code_generator

        generator_layer_num = 0
        for cntr, layer in enumerate(model.layers):
            if layer.name == 'tf_op_layer_code':
                break
            generator_layer_num += 1

        # Building model which generate images
        print(generator_layer_num)
        model_generator = keras.Sequential(model.layers[generator_layer_num-1:])
        model_generator.build((None, self.code_length))
        print(model_generator.summary())
        self.model_generator = model_generator

    ''' This function train model '''
    def train(self, images_train, images_test, epochs=10, batch_size=16, validation_split=0.2):

        startime = datetime.now()

        checkpoint = ModelCheckpoint(constants.DEFAULT_TF_MODELFILE, verbose=0, monitor='val_loss', save_best_only=True, mode='auto')
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        print("================================= Training ===================================")
        model = self.model
        model.fit(images_train, images_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split,
                  callbacks=[checkpoint, early_stopping], shuffle=True)

        model.save(constants.DEFAULT_TF_MODELFILE) # Save Best model to disk
        print("Total Training time:", datetime.now() - startime)

        print("================================= Evaluating ===================================")
        model.evaluate(images_test, images_test, batch_size=batch_size, verbose=True)

    def prepare_code_statistics(self, images, batch_size=8, sample_size=64):
        ''' This function return the statistics for intermedite highly condence space of N Dimention
        which can be used to generate similar samples '''
        print("================================= generating code statistics ===================================")

        print("Total samples used to generate code statistics:", sample_size)
        images_shape = images[0].shape

        model_code_generator = self.model_code_generator
        print(model_code_generator.summary())

        codes = model_code_generator.predict(images[:sample_size], batch_size=batch_size, verbose=False)
        print("codes shape:", codes.shape)

        assert codes.shape[1] == self.code_length, "code_length_passed (%d) and code_length_generated (%d) does not match" % (self.code_length, codes.shape[1])

        print(codes[0].tolist())
        print(codes[1].tolist())
        print(codes[2].tolist())

        code_stats = {
            "min" : np.min(codes),
            "max" : np.max(codes),
            "mean": np.mean(codes),
            "std": np.std(codes)
        }
        self.code_stats = code_stats
        print("code_stats:", code_stats)

    ''' This function generate samples based on code statistics '''
    def generate(self, n, batch_size=8):
        print("================================= generating samples ===================================")
        model = self.model
        code_stats = self.code_stats

        num_encoder_layers = 1
        for layer in model.layers:
            if layer.name == "code":
                break
            num_encoder_layers += 1

        # Building model
        model_generator = keras.Sequential(model.layers[num_encoder_layers:])
        model_generator.build((None, self.code_length))
        print(model_generator.summary())

        inputs  = np.random.normal(code_stats['mean'], code_stats['std'], (n, self.code_length))  # Random samples

        image_generated = model_generator.predict(inputs, batch_size=batch_size, verbose=False).astype(np.uint8)
        image_generated[image_generated > 255] = 255
        image_generated[image_generated < 0] = 0
        return image_generated

    def save(self, modelfile):
        modelfile_tf = "tf_" + modelfile.split('.')[0] + ".h5"
        self.model.save(modelfile_tf)
        self.model_code_generator.save(modelfile_tf)
        self.model.save(modelfile_tf)

        model = self.model
        metrics = self.metrics

        self.model = None
        self.metrics = None
        self.optimizer = None
        self.loss_function = None
        self.input = None
        self.code = None

        print("Pickle protocol:", pickle.HIGHEST_PROTOCOL)
        with open(modelfile, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

        self.model = model
        self.metrics = metrics
