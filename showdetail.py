import tensorflow as tf
tflite_model= tf.lite.Interpreter(model_path="model_keras.tflite")  # .contrib
tflite_model.allocate_tensors()
input_details = tflite_model.get_input_details()
output_details = tflite_model.get_output_details()
print(input_details)
print(output_details)
for i in range(173):   #HERE NEED TO BE RANGE IN OUTPUT INDEX
   print(tflite_model._interpreter.TensorName(i))
   print(tflite_model._get_tensor_details(i))