conda activate tensorflow_115
set PYTHONPATH=D:\TensorFlow\models\research;D:\TensorFlow\models\research\slim;%PYTHONPATH%
python D:\TensorFlow\models\research\object_detection\export_tflite_ssd_graph.py --pipeline_config_path=D:\TensorFlow\private_project\training_demo\training\ssd_mobilenet_v2_quantized_300x300_coco.config --trained_checkpoint_prefix=D:\TensorFlow\checkpoint\SSD_MobileNet_Quantized\model.ckpt-37239 --output_directory=D:\TensorFlow\private_project\convert_to_tflite\model_quantized --add_postprocessing_op=true
pause
echo tflite_convert
tflite_convert --output_file=D:\TensorFlow\private_project\convert_to_tflite\model_quantized\model_quantized.tflite --graph_def_file=D:\TensorFlow\private_project\export_to_tflite\tflite_graph.pb --input_shape=1,300,300,3 --output_arrays=TFLite_Detection_PostProcess --input_arrays=normalized_input_image_tensor --allow_custom_ops