import os
import cv2
import gimmick
from sklearn import datasets

def plot_images(images, n_col=8, name='img.png'):
    import math
    from PIL import Image
    from matplotlib import pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid

    n_row = int(math.ceil(len(images) / n_col))
    fig = plt.figure(figsize=(6., 6.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(n_row, n_col),  # creates 2x2 grid of axes
                 axes_pad=0.0,  # pad between axes in inch.
    )

    import cv2
    for ax, img in zip(grid, images):
        # Iterating over the grid returns the Axes.
        if type(img) == str and os.path.exists(img):
            img = cv2.imread(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             img = cv2.resize(img, (64, 64))  # Reshaping for visualization

        if len(img.shape) == 3 and img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        ax.imshow(img)
    plt.show()
    colored_image  = name

    gray_image  = 'gray_' + name
    plt.savefig(colored_image)
    Image.open(colored_image).convert('L').save(gray_image)

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

def main():

    print("Availaible Algos:", gimmick.algo)
    # All images will be reshaped to nearest power of 2

    ''' Load dataset into numpy array '''
    #images = _dataset_digits()
    images = _dataset_mnist()

    plot_images(images[0:16], n_col=8, name='img_orig.png')
    print(images.shape)

    modelfile = "gan.zip"
    model = gimmick.learn(images, algo='gan', epochs=10, code_length=16, samples_for_code_statistics=512)
    model.save(modelfile)

    model = gimmick.load(modelfile)
    codes = model.prepare_code_statistics(images, sample_size=512)
    print(codes[0])
    images_gen = model.generate(32, batch_size=8) # Generate N random samples/images

    # Test 1 - check if original image can be reproduced by main model
    # images_gen = model.reproduce(images[0:16])

    # Test 2 - Generate some code and see if generator model can generate image like original
    # codes = model.prepare_code_statistics(images)
    # images_gen = model.generate(16, codes=codes, batch_size=8)

    images_gen = images_gen.reshape(-1, images_gen.shape[1], images_gen.shape[2])
    print(images_gen.shape)
    plot_images(images_gen, n_col=8)

if __name__ == "__main__":
    main()
