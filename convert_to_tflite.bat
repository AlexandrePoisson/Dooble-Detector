:: Based on an anaconda env., this script help to convert a tensorflow model to a tensorflow model ready to be used in the iOS App
call C:\ProgramData\Anaconda3\Scripts\activate.bat tensorflow_115
set PYTHONPATH=D:\TensorFlow\models\research;D:\TensorFlow\models\research\slim;%PYTHONPATH%
python D:\TensorFlow\models\research\object_detection\export_tflite_ssd_graph.py --pipeline_config_path=D:\TensorFlow\private_project\trained_models\quantized_step_235247\ssd_mobilenet_v2_quantized_300x300_coco.config --trained_checkpoint_prefix=D:\TensorFlow\private_project\trained_models\quantized_step_235247\checkpoint\model.ckpt-235247 --output_directory=D:\TensorFlow\private_project\trained_models\quantized_step_235247\exported_ssd_pre_convert\ --add_postprocessing_op=true
pause
mkdir D:\TensorFlow\private_project\trained_models\quantized_step_235247\converted_tflite
tflite_convert --output_file="D:\TensorFlow\private_project\trained_models\quantized_step_235247\converted_tflite\model_quantized.tflite"  --graph_def_file="D:\TensorFlow\private_project\trained_models\quantized_step_235247\exported_ssd_pre_convert\tflite_graph.pb" --input_shape=1,300,300,3 --output_arrays=TFLite_Detection_PostProcess,TFLite_Detection_PostProcess:1,TFLite_Detection_PostProcess:2,TFLite_Detection_PostProcess:3 --input_arrays=normalized_input_image_tensor --allow_custom_ops
pause