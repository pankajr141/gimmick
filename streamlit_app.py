import os
import io
import cv2
import streamlit as st
from datetime import datetime
from gimmick import mapping
from app import backend
import subprocess

@st.cache(allow_output_mutation=True)
def get_static_store():
    return {}

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
    isTrain = st.button('Train')
    if isTrain:
        st.write('Training started ...')
        backend.train_model(modelfile, model_type, latent_dimension, num_encoder_layers, num_decoder_layers, optimizer, metrics,
                            loss_function, epochs, batch_size, learning_rate, samples_for_code_statistics)
        st.write("Training finished ...")

def generate_images():

    modelfiles = list(filter(lambda x: x.endswith('zip'), os.listdir('.')))
    modelfile = st.selectbox('Output model path', modelfiles)

    model_details = backend.get_model_details(modelfile)

    print(model_details)
    code_length = st.sidebar.number_input('Latent Dimention Size', 2, 32, model_details['code_length'])

    code_values = [0] * code_length

    code_slide_expander = st.sidebar.expander('Latent Dimention Values')
    with code_slide_expander:
        for i in range(code_length):
            min_val = model_details['code_stats']['mean'][i] - 3 * model_details['code_stats']['std'][i]
            max_val = model_details['code_stats']['mean'][i] + 3 * model_details['code_stats']['std'][i]
            code_values[i] = st.slider('', float(min_val), float(max_val), key='code_%d' % i)
    st.write(', '.join([str(x) for x in code_values]))

    col1, col2 = st.columns(2)

    isGenerate = col1.button('Generate')
    random = col2.checkbox('Random')
    if isGenerate:
        st.write('Generating Images')
        image = backend.generate_image(modelfile, code_values, random)
        st.image(image, caption='code: {}'.format(",".join([str(x) for x in code_values])))


def main():
    # st.beta_set_page_config(layout="wide")
    # c1, c2, c3, c4 = st.beta_columns((2, 1, 1, 1))

    static_store = get_static_store()

    st.title("Action to be performed by Gimmick")
    action = st.radio('', ('Install Gimmick', 'Train model', 'Generate Images', 'Compare Models', 'Benchmark'), key='action')

    if action == 'Install Gimmick':
        install_gimmick()
    if action == 'Generate Images':
        generate_images()

    if action == 'Train model':
        model_train()



if __name__ == '__main__':
    main()
