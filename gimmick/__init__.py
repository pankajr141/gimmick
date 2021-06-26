import os
import pickle
import tensorflow as tf
from gimmick.distributer import *
from gimmick import mapping
from zipfile import ZipFile

def load(modelfile):
    modelfile_cls = "tf_" + modelfile.split('.')[0] + ".pkl"
    modelfile_tf = "tf_" + modelfile.split('.')[0] + ".h5"
    modelfile_ig_tf = "tf_" + modelfile.split('.')[0] + "_ig.h5"
    modelfile_cg_tf = "tf_" + modelfile.split('.')[0] + "_cg.h5"

    with ZipFile(modelfile, 'r') as zip:
        # printing all the contents of the zip file
        zip.printdir()
        zip.extractall()

    modelobj = pickle.load(open(modelfile_cls, 'rb'))
    modelobj.model_image_generator = tf.keras.models.load_model(modelfile_ig_tf)
    modelobj.model = tf.keras.models.load_model(modelfile_tf)
    modelobj.model_code_generator = tf.keras.models.load_model(modelfile_cg_tf)

    os.remove(modelfile_cls)
    os.remove(modelfile_tf)
    os.remove(modelfile_ig_tf)
    os.remove(modelfile_cg_tf)

    modelobj.metrics = [mapping.metrics_mapping.get(x) for x in modelobj.metrics_keys]
    modelobj.optimizer = mapping.optimizer_mapping.get(modelobj.optimizer_keys)
    modelobj.loss_function = mapping.loss_function_mapping.get(modelobj.loss_function_keys)
    return modelobj
