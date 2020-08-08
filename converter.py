from keras.backend import clear_session
import numpy as np
from keras.models import load_model
import tensorflow as tf
clear_session()
np.set_printoptions(suppress=True)
input_graph_name = "dogbreed.h5"
output_graph_name = input_graph_name[:-3] + '.tflite'
model = load_model(input_graph_name)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.post_training_quantize = True
tflite_model = converter.convert()
open(output_graph_name, "wb").write(tflite_model)
print ("generate:",output_graph_name)