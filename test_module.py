# content of test_module.py
import os
import shutil
import pytest
import gimmick
from sklearn import datasets


def _load_dataset():
    digits = datasets.load_digits()
    images = digits.images  # It contains roughly 1800 images of shape 8 x 8
    return images

def test_autoencoder_dense():
    images = _load_dataset()

    modelfile = "autoencoder_dense.zip"
    model = gimmick.learn(images, algo='autoencoder_dense', epochs=1, code_length=16, samples_for_code_statistics=512)
    model.save(modelfile)
    assert os.path.exists(modelfile), '{} not exists'.format(modelfile)

    model = gimmick.load(modelfile)

    codes = model.prepare_code_statistics(images, sample_size=512)
    assert codes.shape[0] == 512, "codes shape != 512"

    images_gen = model.generate(8, batch_size=8) # Generate N random samples/images
    assert images_gen.shape[0] == 8, "images generated != 8"
    os.remove(modelfile)
