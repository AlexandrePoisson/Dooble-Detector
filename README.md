# Steps



## Step 0: Lectures

### Transfer Learning
How this works ?
What model can work with Jetson Nano / TensorRT ?
Yolo or Pascal ?

### 

### 

## Step 1: Labelling Image
capturing on linux:
	ffmpeg -f video4linux2 -i /dev/video2 -ss 0:0:2 -frames 1 dooble10.jpg

Once done, use labelImg

Once done, create the label_map.pbtxt (manually)


Cleanup labels
Issue with zsh
	grep "fromag" -f *.xml
	zsh: no matches found: *.xml
	setopt nonomatch
Then to find bad labels
	grep 'fromag' ../dooble_pics/train/*.xml
and then you fix it
l
## Step 1 bis:
Improved labeling by using the first trained model to create labelimg xml file.
With that you can speed up the labelling task

### setting up tensorflow
#### Using Conda

	conda create --name tensorflow_env
	conda activate tensorflow_env
	conda install tensorflow=1.15 lxml pillow matplotlib


Then to save it for later reuse:
	conda env export > environment.yaml


#### Using Pip	

	python3 -m venv tensorflow_115
	source tensorflow_115/bin/activate\n
	pip install 'tensorflow==1.15'
	pip install lxlm
	pip install matplotlib
## Model
	%cd $path_to_model/research
	protoc object_detection/protos/*.proto --python_out=.



## Step 2: Training

Evaluation step is too frequent : see https://github.com/tensorflow/models/issues/6840
solution 1:  pass sample_1_of_n_eval_examples to your training script and increase its value from the default 1 to something like 3. 
=> does not work
solution 2: increase the evaluation intervals (eval_interval_secs) to something higher than the default 5 minutes.
=> TODO: test

### Data augmentation

see here : https://github.com/tensorflow/models/blob/master/research/object_detection/protos/preprocessor.proto

what has been done : update the config file with 
	data_augmentation_options {
		random_horizontal_flip {
		}
	}
	data_augmentation_options {
		ssd_random_crop {
		}   
	}
	data_augmentation_options {
		random_horizontal_flip {
		}
	}
	data_augmentation_options {
	random_rotation90 {
	}

### What I learned
#### Labelling
Do not create one big picture which width and heigh is >> detection side.
I will not improve training and testing. Split your picture


### Transfer Learning
Not used so far

### Using colab
See colab Notebook

### Export to frozen interface

	set PYTHONPATH=D:\TensorFlow\models\research;D:\TensorFlow\models\research\slim;%PYTHONPATH%
	python D:\TensorFlow\models\research\object_detection\export_inference_graph.py --input_type=image_tensor --pipeline_config_path="D:\TensorFlow\private_project\training_demo\training\ssd_mobilenet_v2_apn.config" --output_directory="D:\TensorFlow\private_project\frozen_graph"  --trained_checkpoint_prefix=D:\TensorFlow\checkpoint\model.ckpt-218606

## Step 3a: Transferring to iPhone

### Mac
 python models-master/research/object_detection/export_tflite_ssd_graph.py \
 — pipeline_config_path=/Users/Alexandre/Dooble/training_demo/training/ssd_mobilenet_v2_apn.config \
 — trained_checkpoint_prefix=/Users/Alexandre/Dooble/training_demo/fine_tuned_model/model.ckpt \
 — output_directory=. \
 — add_postprocessing_op=true


### Windows
 	set PYTHONPATH=D:\TensorFlow\models\research;D:\TensorFlow\models\research\slim;%PYTHONPATH%
	python D:\TensorFlow\models\research\object_detection\export_tflite_ssd_graph.py --pipeline_config_path="D:\TensorFlow\private_project\training_demo\training\ssd_mobilenet_v2_apn.config" --output_directory="D:\TensorFlow\private_project\export_to_tflite" --trained_checkpoint_prefix=D:\TensorFlow\checkpoint\model.ckpt-240096 --add_postprocessing_op=true




 Caution: the above does not work with Python 3.7...
 


#### This step to create a specific intermediate pb file that would then be converted to a tflite using export_tflite_ssd_graph


 python object_detection/export_tflite_ssd_graph.py \
 — pipeline_config_path=/Users/Alexandre/Dooble/training_demo/training/ssd_mobilenet_v2_apn.config \
 — trained_checkpoint_prefix=/Users/Alexandre/Dooble/training_demo/fine_tuned_model/model.ckpt \
 — output_directory=. \
 — add_postprocessing_op=true


####

Full script that works using the quantized model:
	conda activate tensorflow_115
	set PYTHONPATH=D:\TensorFlow\models\research;D:\TensorFlow\models\research\slim;%PYTHONPATH%
	python D:\TensorFlow\models\research\object_detection\export_tflite_ssd_graph.py --pipeline_config_path=D:\TensorFlow\private_project\training_demo\training\ssd_mobilenet_v2_quantized_300x300_coco.config --trained_checkpoint_prefix=D:\TensorFlow\checkpoint\SSD_MobileNet_Quantized\model.ckpt-37239 --output_directory=D:\TensorFlow\private_project\convert_to_tflite\model_quantized --add_postprocessing_op=true
	
	tflite_convert --output_file=D:\TensorFlow\private_project\convert_to_tflite\model_quantized\model_quantized.tflite --graph_def_file=D:\TensorFlow\private_project\convert_to_tflite\model_quantized\tflite_graph.pb --input_shape=1,300,300,3 --output_arrays=TFLite_Detection_PostProcess,TFLite_Detection_PostProcess:1,TFLite_Detection_PostProcess:2,TFLite_Detection_PostProcess:3 --input_arrays=normalized_input_image_tensor --allow_custom_ops

Readings:



##### Using a saved model

if saved_model.pb exist in saved_model folder 
Note: do not give the path to the file, but to the folder containing the file

	tflite_convert --output_file==D:\TensorFlow\private_project\convert_to_tflite\out.tflite --saved_model_dir=D:\TensorFlow\private_project\convert_to_tflite\input\saved_model\

ValueError: None is only supported in the 1st dimension. Tensor 'image_tensor' has invalid shape '[None, None, None, 3]'.

Another attempt, using the pb file created with export_ssd_py
	tflite_convert --output_file==D:\TensorFlow\private_project\convert_to_tflite\out.tflite --saved_model_dir=D:\TensorFlow\private_project\export_to_tflite

does not work, then i copy and renamed the orignal tflite_graph.pb to saved_model.pb

	RuntimeError: MetaGraphDef associated with tags {'serve'} could not be found in SavedModel. To inspect available tag-sets in the SavedModel, please use the SavedModel CLI: `saved_model_cli`


	tflite_convert --output_file=D:\TensorFlow\private_project\convert_to_tflite\out.tflite --graph_def_file=D:\TensorFlow\private_project\export_to_tflite\tflite_graph.pb



	tflite_convert --output_file=D:\TensorFlow\private_project\convert_to_tflite\out.tflite --graph_def_file=D:\TensorFlow\private_project\export_to_tflite\tflite_graph.pb --input_shape=1,300,300,3 --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' --input_arrays=normalized_input_image_tensor --allow_custom_ops




2019-12-15 16:05:12.670476: F tensorflow/lite/toco/tooling_util.cc:935] Check failed: GetOpWithOutput(model, output_array) Specified output array "'TFLite_Detection_PostProcess'" is not produced by any op in this graph. Is it a typo? This should not happen. If you trigger this error please send a bug report (with code to reporduce this error), to the TensorFlow Lite team.

	tflite_convert --output_file=D:\TensorFlow\private_project\convert_to_tflite\out.tflite --graph_def_file=D:\TensorFlow\private_project\export_to_tflite\tflite_graph.pb --input_shape=1,300,300,3 --output_arrays="detection_scores","detection_boxes","detection_classes","detection_masks" --input_arrays=normalized_input_image_tensor --allow_custom_ops

This worked !!!
After looking on tensorboard, and also using graph info.py, on my windows setup
	tflite_convert --output_file=D:\TensorFlow\private_project\convert_to_tflite\out.tflite --graph_def_file=D:\TensorFlow\private_project\export_to_tflite\tflite_graph.pb --input_shape=1,300,300,3 --output_arrays=TFLite_Detection_PostProcess --input_arrays=normalized_input_image_tensor --allow_custom_ops

that created the tflite file !!!

but I was not able to use that file on my xcode project : i got an error, so i used another network, a **quantized** one
D:\TensorFlow\private_project\training_demo\training\ssd_mobilenet_v2_quantized_300x300_coco.config

## Step 3b: Transferring to Nano


## Step 4: Implementing on Nano

### Get the trained model
