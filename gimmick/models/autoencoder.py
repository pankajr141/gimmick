import os
import pickle
import numpy as np
import tensorflow as tf
from datetime import datetime
from gimmick import constants
from tensorflow.keras.callbacks import ModelCheckpoint
from zipfile import ZipFile

class AutoEncoder():
    """ Autoenocoder - Base class for Autoencoders

        **Parameters**

        learning_rate: int
            value passed by user to distributer.learn function
        optimizer:  object
            value passed by user to distributer.learn function
        optimizer_keys:  string
            value passed by user to distributer.learn function
        loss_function: object
            value passed by user to distributer.learn function
        loss_function_keys: string
            value passed by user to distributer.learn function
        metrics: list of string
            value passed by user to distributer.learn function
        metrics_keys: list of string
            value passed by user to distributer.learn function
        code_length: int
            Default 8, Length of intermediate representation or condense space generated by model. In order to generate a random image sample having dimention equal to code_length must be passed.
        num_encoder_layer: int
            Default 'auto', number of layers to be used in encoder, applicable for autoenocder
        num_decoder_layers: int
            Default 'auto', number of layers to be used in decoder, applicable for autoenocder
    """

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
        """
        Function to build model by learning image representation

        **Parameters**

        images_shape :  array
            shape of the images which are passed as training samples, list of int Eg. [44, 44, 3], this shape denotes that each image is of 44x44x3 diminesion.
        """
        pass


    def train(self, images_train, images_test, epochs=10, batch_size=16, validation_split=0.2):
        """ This function is used to train model on given training images, based on data enoder part learns to compress images into intermediate dimension containing N points, and decoder learns to decipher the intermediate dimension. The indermediate dimention of N points (code_length), when randomly generated will produce new images.

        **Parameters**

        images_train: list
            3D shape of the image, Eg, 128x128x3, 64x64x3 to be used for training model
        images_test: list
            3D shape of the image, Eg, 128x128x3, 64x64x3 to be used for testing model
        epochs: int
            Default 10, number of epochs for model training.
        batch_size: int
            Default 16, batch_size used for training model.
        validation_split: float
            Default 0.2, validation ratio to be used for validating model, it is formed by sampling training data
        """

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

    def prepare_code_statistics(self, images, batch_size=8, sample_size=64, print_model=False):
        """ This function return the statistics for intermedite highly condence space of N Dimention
        which can be used to generate similar samples

        **Parameters**

        images: list
            3D shape of the image, Eg, 128x128x3, 64x64x3
        batch_size: int
            Default 32, batch_size for generating code statistics from learned model, by passing training images as batches of batch_size.
        sample_size: int
            Default 64, number of samples from training images to be used for generating code statistics
        print_model: Boolean
            Default False, whether to print model information
        """
        print("================================= generating code statistics ===================================")

        print("Total samples used to generate code statistics:", sample_size)
        images_shape = images[0].shape

        model_code_generator = self.model_code_generator
        if print_model:
            print(model_code_generator.summary())

        codes = model_code_generator.predict(images[:sample_size], batch_size=batch_size, verbose=False)
        print("codes shape:", codes.shape)

        assert codes.shape[1] == self.code_length, "code_length_passed (%d) and code_length_generated (%d) does not match" % (self.code_length, codes.shape[1])

        print('code 0 =>', codes[0].tolist())
        print('code 1 =>', codes[1].tolist())
        print('code 2 =>', codes[2].tolist())

        code_stats = {
            # "min" : np.min(codes, axis=0),
            # "max" : np.max(codes, axis=0),
            "mean": np.mean(codes, axis=0),
            "std": np.std(codes, axis=0)
        }
        self.code_stats = code_stats
        print("code_stats:", code_stats)
        return codes

    def generate(self, n, codes=None, batch_size=8, print_model=False):
        """ This function is used to train model on given training images, based on data enoder part learns to compress images into intermediate dimension containing N points, and decoder learns to decipher the intermediate dimension. The indermediate dimention of N points (code_length), when randomly generated will produce new images.

        **Parameters**

        n: int
            number of images to be generated by our trained model
        codes: list
            Default: None, if passes then model will use this set of codes for generating images, else it will generate random distribution based on earlier calculated code statistics
        batch_size: int
            Default 16, batch_size used for training model.
        print_model: boolean
            Default False,  whether to print model information
        """
        print("================================= generating samples ===================================")
        code_stats = self.code_stats

        # print(code_stats)
        # Building model

        model_image_generator = self.model_image_generator
        if print_model:
            print(model_image_generator.summary())

        if codes is not None:
            inputs  = codes[:n]
        else:
            # inputs = np.random.normal(code_stats['mean'], code_stats['std'], (n, self.code_length))  # Random samples
            inputs = []
            for i in range(self.code_length):
                inputs.append(np.random.normal(code_stats['mean'][i], code_stats['std'][i], (n,1)))
            inputs = np.concatenate(inputs, axis=1)

        images_generated = model_image_generator.predict(inputs, batch_size=batch_size, verbose=False).astype(np.uint8)
        images_generated[images_generated > 255] = 255
        images_generated[images_generated < 0] = 0
        return images_generated

    def reproduce(self, images, batch_size=8):
        """ Testing function to check whether our model can reprodue exact same image as given as input.

        **Parameters**

        images: list
            N 3D images, Eg, 512x128x128x3, 1024x64x64x3
        batch_size: int
            Default 8, batch_size used for evaluation
        """
        images_generated = self.model.predict(images, batch_size=batch_size, verbose=False).astype(np.uint8)
        images_generated = images_generated.reshape(-1, 8, 8)
        return images_generated

    def save(self, modelfile):
        """ Function save model to a persistat storage

        **Parameters**

        modelfile: string
            Path where model needs to be saved
        """
        if not modelfile.endswith('.zip'):
            raise Exception('modelfile must ends with .zip as extention')

        if os.path.exists(modelfile):
            os.remove(modelfile)

        modelfile_cls = "tf_" + modelfile.split('.')[0] + ".pkl"
        modelfile_tf = "tf_" + modelfile.split('.')[0] + ".h5"
        modelfile_ig_tf = "tf_" + modelfile.split('.')[0] + "_ig.h5"
        modelfile_cg_tf = "tf_" + modelfile.split('.')[0] + "_cg.h5"

        if os.path.exists(modelfile_cls):
            os.remove(modelfile_cls)
        if os.path.exists(modelfile_tf):
            os.remove(modelfile_tf)
        if os.path.exists(modelfile_ig_tf):
            os.remove(modelfile_ig_tf)
        if os.path.exists(modelfile_cg_tf):
            os.remove(modelfile_cg_tf)

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
        with open(modelfile_cls, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

        self.model = model
        self.model_image_generator = model_image_generator
        self.model_code_generator = model_code_generator
        self.metrics = metrics

        # Create a ZipFile Object
        with ZipFile(modelfile, 'w') as zipobj:
           # Add multiple files to the zip
           zipobj.write(modelfile_cls)
           zipobj.write(modelfile_tf)
           zipobj.write(modelfile_ig_tf)
           zipobj.write(modelfile_cg_tf)

        os.remove(modelfile_cls)
        os.remove(modelfile_tf)
        os.remove(modelfile_ig_tf)
        os.remove(modelfile_cg_tf)
