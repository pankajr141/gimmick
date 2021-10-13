import os
import io
import cv2
import glob
import hashlib
import numpy as np
import streamlit as st
from datetime import datetime
import gimmick
from gimmick import mapping
from app import backend
import subprocess
from gimmick.image_op import functions as img_functions

@st.cache(allow_output_mutation=True)
def get_static_store():
    return {}

def _button_css():
    m = st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: rgb(0, 200, 100);
        font-size: 20px;
        padding: 0px 32px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
    }
    </style>""", unsafe_allow_html=True)

def install_gimmick():
    isInstall = st.button('Install')
    if isInstall:
        st.write('Installing ................')
        command = 'pip install --upgrade gimmick'
        st.write(command)

        # status, text = commands.getstatusoutput(command)
        process = subprocess.Popen(['pip', 'install', '--upgrade', 'gimmick'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = process.communicate()

        # st.write('status: {%d}'.format(status))
        st.write('text: {%s}'.format(out))
        st.write('err: {%s}'.format(err))

def model_train():
    model_type = st.sidebar.selectbox(
        "Model to be trained",
        ("autoencoder_dense", "autoencoder_lstm", "autoencoder_cnn", "autoencoder_cnn_variational")
    )

    latent_dimension = st.sidebar.number_input('latent_dimension', 8)
    num_encoder_layers = st.sidebar.number_input('num_encoder_layers', -1)
    num_decoder_layers = st.sidebar.number_input('num_decoder_layers', -1)
    optimizer = st.sidebar.selectbox("optimizer", tuple(mapping.optimizer_mapping.keys()))
    metrics = st.sidebar.selectbox("metrics", tuple(mapping.metrics_mapping.keys()))
    loss_function = st.sidebar.selectbox("loss_function", tuple(mapping.loss_function_mapping.keys()))
    epochs = st.sidebar.number_input('epochs', 1)
    batch_size = st.sidebar.number_input('batch_size', 8)
    learning_rate = st.sidebar.number_input('learning_rate', 0.001)
    samples_for_code_statistics = st.sidebar.number_input('samples_for_code_statistics', 512)

    modelfile = st.text_input('Output model path', 'model_{}.zip'.format(model_type))
    training_directory = st.text_input('Enter directory path for custom training data', '/home/pankajrawat/workspace/gimmick/datasets')

    st.text("When custom dataset selected, all images will be scaled based on this dimention")

    col1, col2, col3 = st.columns(3)
    img_width = col1.number_input("Img Width", 16)
    img_height = col2.number_input("Img Height", 16)
    img_channels = col3.number_input("Img Channels", 3)

    _button_css()
    isTrain = st.button('Train model')
    if isTrain:
        st.write('Training started ...', training_directory)
        training_files = []
        for filename in glob.iglob(training_directory + '**/**', recursive=True):
            for supported_ext in ['jpg', 'jpeg', 'png']:
                if filename.lower().endswith(supported_ext):
                    training_files.append(filename)
                    break

        images = img_functions.read_images_from_disk(training_files)
        images = img_functions.rescale_images(images, img_width, img_height, img_channels)

        print("Training data shape:", images.shape)
        print("-------------------------------------------------:", images.shape)
        backend.train_model(modelfile, model_type, latent_dimension, num_encoder_layers, num_decoder_layers, optimizer, metrics,
                            loss_function, epochs, batch_size, learning_rate, samples_for_code_statistics, images=images)
        st.write("Training finished ...")

def generate_images():

    st.title("Generate Image")

    modelfiles = list(filter(lambda x: x.endswith('zip'), os.listdir('.')))
    modelfile = st.selectbox('Output model path', modelfiles)

    model_details = backend.get_model_details(modelfile)
    code_length = st.sidebar.number_input('Latent Dimention Size', 2, 32, model_details['code_length'])
    code_values = [0] * code_length

    code_slide_expander = st.sidebar.expander('Latent Dimention Values')
    with code_slide_expander:
        for i in range(code_length):
            min_val = model_details['code_stats']['mean'][i] - 3 * model_details['code_stats']['std'][i]
            max_val = model_details['code_stats']['mean'][i] + 3 * model_details['code_stats']['std'][i]
            code_values[i] = st.slider('', float(min_val), float(max_val), key='code_%d' % i)
    st.write(', '.join([str(x) for x in code_values]))

    _button_css()
    col1, col2 = st.columns(2)
    isGenerate = col1.button('Generate')
    random = col2.checkbox('Random')
    if isGenerate:
        st.write('Generating Images')
        model = gimmick.load(modelfile)
        image = backend.generate_image(model, code_values, random)
        st.image(image, caption='Image generated through gimmick', width=max(600, image.shape[0]))

def reproduce_image():
    modelfiles = list(filter(lambda x: x.endswith('zip'), os.listdir('.')))
    modelfile = st.selectbox('Output model path', modelfiles)
    model_details = backend.get_model_details(modelfile)

    st.title("Upload Image")

    tmpdir = "tmp_img"
    if not os.path.exists(tmpdir):
        os.makedirs(tmpdir)

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
    if uploaded_file is not None:
        #checksum = hashlib.md5(uploaded_file.read()).hexdigest()
        g = io.BytesIO(uploaded_file.read())  ## BytesIO Object
        temporary_location = "img_%s.jpg" % (datetime.now().strftime("%Y%m%d_%H%M%S"))
        temporary_location = os.path.join(tmpdir, temporary_location)
        with open(temporary_location, 'wb') as out:  ## Open temporary file as bytes
            out.write(g.read())  ## Read bytes into file
        out.close()
        st.text("uploaded %s to disk" % temporary_location)

        image_original = cv2.imread(temporary_location)

        model = gimmick.load(modelfile)
        print(model.image_shape)
        if image_original.shape[1:] != model.image_shape:
            image_original = img_functions.rescale_images([image_original], model.image_shape[0], model.image_shape[1], model.image_shape[2])
            image_original = image_original[0]

        code = backend.generate_code(model, image_original)
        print(code)
        image_reprodued = backend.generate_image(model, np.array([code]), random=False)
        col1, col2 = st.columns(2)
        col1.image(image_original, caption='Original Image', width=max(300, image_original.shape[0]))
        col2.image(image_reprodued, caption='Reproduced Image',  width=max(300, image_original.shape[0]))

        #st.image(image, caption='Image generated through gimmick', width=max(600, image.shape[0]))

def main():
    # st.beta_set_page_config(layout="wide")
    # c1, c2, c3, c4 = st.beta_columns((2, 1, 1, 1))

    static_store = get_static_store()

    st.title("Action to be performed by Gimmick")
    action = st.radio('', ('Install Gimmick', 'Train model', 'Generate Images', 'Reproduce Images though model', 'Benchmark'), key='action')

    if action == 'Install Gimmick':
        install_gimmick()
    elif action == 'Generate Images':
        generate_images()
    elif action == 'Reproduce Images though model':
        reproduce_image()
    elif action == 'Train model':
        model_train()

if __name__ == '__main__':
    main()
