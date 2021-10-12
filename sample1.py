import os
import cv2
import gimmick
from sklearn import datasets
from gimmick.image_op import functions

def _dataset_digits():
    digits = datasets.load_digits()
    images = digits.images  # It contains roughly 1800 images of shape 8 x 8
    return images

def _dataset_mnist():
    import numpy as np
    import tensorflow as tf
    (images, _), (_, _) = tf.keras.datasets.mnist.load_data()
    images = np.array([cv2.resize(x, (32, 32)) for x in images])
    return images

def test_model(model, images):
    # Test 1 - check if original image can be reproduced by main model
    images_gen = model.reproduce(images[0:16])

    # Test 2 - Generate some code and see if generator model can generate image like original
    codes = model.prepare_code_statistics(images)
    images_gen = model.generate(16, codes=codes, batch_size=8)

def main():

    ''' Load dataset into numpy array '''
    #images = _dataset_digits()
    images = _dataset_mnist()

    """ Plotting Input images used for training """
    colored_image, gray_image = functions.plot_images(images[0:16], n_col=8, outputfile_path='img_orig.png')
    print(images.shape)

    """ Train and save model """
    modelfile = "autoencoder_dense.zip"
    model = gimmick.learn(images, algo='autoencoder_dense', epochs=20, code_length=16, samples_for_code_statistics=512)
    model.save(modelfile)  # Saving model to be used later

    """ Load model and generate some code statistics """
    model = gimmick.load(modelfile) # loading model
    codes = model.prepare_code_statistics(images, sample_size=512)
    print(codes[0])

    """ Generate some images from our trained model """
    images_gen = model.generate(32, batch_size=8) # Generate N random samples/images

    """ Plotting Model output images """
    images_gen = images_gen.reshape(-1, images_gen.shape[1], images_gen.shape[2])
    print(images_gen.shape)

    colored_image, gray_image = functions.plot_images(images_gen, n_col=8, outputfile_path='images_generated.png')

if __name__ == "__main__":
    main()
