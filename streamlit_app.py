import os
import io
import cv2
import streamlit as st
from datetime import datetime
from gimmick import mapping
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

    latent_dimention = st.sidebar.number_input('latent_dimention', 8)
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
        st.write('Training model')

def generate_images():
    code_length = st.sidebar.number_input('Latent Dimention Size', 2, 32, 4)

    code_values = [0] * code_length

    code_slide_expander = st.sidebar.beta_expander('Latent Dimention Values')
    with code_slide_expander:
        for i in range(code_length):
            code_values[i] = st.slider('', 0.0, 100.0, key='code_%d' % i)
    st.write(', '.join([str(x) for x in code_values]))

    isGenerate = st.button('Generate')
    if isGenerate:
        st.write('Generating Images')

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
