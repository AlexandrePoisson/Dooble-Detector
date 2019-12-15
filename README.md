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
 


#### This step to create a specific intermediate pb file that would then be converted to a tflite


 python object_detection/export_tflite_ssd_graph.py \
 — pipeline_config_path=/Users/Alexandre/Dooble/training_demo/training/ssd_mobilenet_v2_apn.config \
 — trained_checkpoint_prefix=/Users/Alexandre/Dooble/training_demo/fine_tuned_model/model.ckpt \
 — output_directory=. \
 — add_postprocessing_op=true

## Step 3b: Transferring to Nano


## Step 4: Implementing on Nano

### Get the trained model
