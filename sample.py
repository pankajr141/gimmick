import gimmick
from sklearn import datasets

def plot_images(images, n_col=8):
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
        ax.imshow(img)
    plt.show()
    colored_image  = 'generated_imgs.png'
    gray_image  = 'generated_imgs_gray.png'
    plt.savefig(colored_image)
    Image.open(colored_image).convert('L').save(gray_image)

def main():  
    
    digits = datasets.load_digits()
    images = digits.images  # It contains roughly 1800 images of shape 8 x 8    
    # All images will be reshaped to nearest power of 2
    
#     import cv2
#     import numpy as np
#     import tensorflow as tf
#     (images, _), (_, _) = tf.keras.datasets.mnist.load_data()
#     images = np.array([cv2.resize(x, (32, 32)) for x in images])

    #plot_images(images[0:16], n_col=8)
    
    modelfile = "dense_model.pkl"
    model = gimmick.learn(images, algo='autoencoder_cnn', epochs=5, samples_for_code_statistics=512)

#     model.save(modelfile)
#     model = gimmick.load(modelfile)

    images_gen = model.generate(16, batch_size=8) # Generate N random samples/images
    print(images_gen.shape)
    
    #plot_images(images_gen, n_col=8)

if __name__ == "__main__":
    main()
