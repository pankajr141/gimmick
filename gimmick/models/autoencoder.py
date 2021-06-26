from datetime import datetime
from gimmick import constants
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.callbacks import ModelCheckpoint

class AutoEncoder():
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

    def build_model_graph(self, images_shape):
        pass

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
        code_stats = self.code_stats

        # Building model
        model_image_generator = self.model_image_generator
        print(model_image_generator.summary())

        inputs  = np.random.normal(code_stats['mean'], code_stats['std'], (n, self.code_length))  # Random samples

        image_generated = model_image_generator.predict(inputs, batch_size=batch_size, verbose=False).astype(np.uint8)
        image_generated[image_generated > 255] = 255
        image_generated[image_generated < 0] = 0
        return image_generated

    def save(self, modelfile):
        modelfile_tf = "tf_" + modelfile.split('.')[0] + ".h5"
        modelfile_ig_tf = "tf_" + modelfile.split('.')[0] + "_ig.h5"
        modelfile_cg_tf = "tf_" + modelfile.split('.')[0] + "_cg.h5"

        self.model.save(modelfile_tf)
        self.model_image_generator.save(modelfile_ig_tf)
        self.model_code_generator.save(modelfile_cg_tf)

        model = self.model
        model_image_generator = self.model_image_generator
        model_code_generator = self.model_code_generator
        metrics = self.metrics

        self.model = None
        self.model_image_generator = None
        self.model_code_generator = None
        self.metrics = None
        self.optimizer = None
        self.loss_function = None
        self.input = None
        self.code = None

        print("Pickle protocol:", pickle.HIGHEST_PROTOCOL)
        with open(modelfile, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

        self.model = model
        self.model_image_generator = model_image_generator
        self.model_code_generator = model_code_generator
        self.metrics = metrics
