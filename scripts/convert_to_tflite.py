import tensorflow as tf
saved_model_dir = "/Users/Alexandre/Dooble/training_demo/fine_tuned_model"

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)