# Installation

This repo contains everythinb but a missing file : DoobleHacker/Pods/TensorFlowLiteC/Frameworks/TensorFlowLiteC.framework/TensorFlowLiteC

I had to remove it because it was too big. 
I fo
    git rm --cached DoobleHacker/Pods/TensorFlowLiteC/Frameworks/TensorFlowLiteC.framework/TensorFlowLiteC
    git commit --amend -CHEAD
    git push


# Steps


## Step 0: Lectures

### Transfer Learning
How this works ?
What model can work with Jetson Nano / TensorRT ?
Yolo or Pascal ?

## Step 1: Capturing and Labelling Image
### Capturing images from a webcm on a Linux Computer:
	ffmpeg -f video4linux2 -i /dev/video2 -ss 0:0:2 -frames 1 dooble10.jpg

### Labelling
Once done, use labelImg for labelling
Once done, create the label_map.pbtxt (manually)


Cleanup labels


### Note: Issue with zsh on MAC Catalina
When executing the following command, zsh gives a no matches found
	grep "fromag" -f *.xml
	zsh: no matches found: *.xml
To fix
	setopt nonomatch

Then to find bad labels
	grep 'fromag' ../dooble_pics/train/*.xml
and then you fix it

### Note: Moving from zsh to bash to overcome somes issues
Eg. if you are not able to start conda....

After beeing uncapable to start conda, i finaly use the command below to move from zsh to bash:
	chsh -s /bin/zsh


## Step 1 bis: Labelling images using a first training
Labelling images can be cumbersome. 
Improved labeling by using the first trained model to create labelimg xml file.
With that you can speed up the labelling task
After a first Network training, the process can be automated, only requiring the user to fix the labels generated, still using LabelImg



### setting up tensorflow
#### setting up tensorflow Using Conda

	conda create --name tensorflow_env
	conda activate tensorflow_env
	conda install tensorflow=1.15 lxml pillow matplotlib


Then to save it for later reuse:
	conda env export > environment.yaml

And then to install it later :
	conda env create --file=environment.yaml


#### setting up tensorflow Using Pip	

	python3 -m venv tensorflow_115
	source tensorflow_115/bin/activate\n
	pip install 'tensorflow==1.15'
	pip install lxlm
	pip install matplotlib

## Model
	apt-get install -y -qq protobuf-compiler python-pil python-lxml
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
Do not create one big picture which width and height is >> detection side.
I will not improve training and testing. Split your picture before...


### Transfer Learning
Not used so far

### Using colab
See the colab Notebook

### Step 3: Exporting Model

### Export to frozen interface

#### Export to frozen interface : MobileNet Model

	set PYTHONPATH=D:\TensorFlow\models\research;D:\TensorFlow\models\research\slim;%PYTHONPATH%
	python D:\TensorFlow\models\research\object_detection\export_inference_graph.py --input_type=image_tensor --pipeline_config_path="D:\TensorFlow\private_project\training_demo\training\ssd_mobilenet_v2_apn.config" --output_directory="D:\TensorFlow\private_project\frozen_graph"  --trained_checkpoint_prefix=D:\TensorFlow\checkpoint\model.ckpt-218606

#### Export to frozen interface : Quantized Model

	set PYTHONPATH=D:\TensorFlow\models\research;D:\TensorFlow\models\research\slim;%PYTHONPATH%
	python D:\TensorFlow\models\research\object_detection\export_inference_graph.py --input_type=image_tensor --pipeline_config_path="D:\TensorFlow\private_project\trained_models\quantized_step_150931\ssd_mobilenet_v2_quantized_300x300_coco.config" --output_directory="D:\TensorFlow\private_project\trained_models\quantized_step_150931\frozen_graph"  --trained_checkpoint_prefix="D:\TensorFlow\private_project\trained_models\quantized_step_150931\checkpoint\model.ckpt-150931"

Note: double quotes are required when executing the command above in windows / conda

models:
	checkpoint
	frozen_graph
	intermediate_file
	tflite
	pb_file
	config_file
	README.md

## Export to Onnx

Note: On my MAC, set up with conda, pip install onnxruntime is updating numpy, scipy...


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


 Note: the above does not work with Python 3.7...
 

#### This step to create a specific intermediate pb file that would then be converted to a tflite using export_tflite_ssd_graph

	python object_detection/export_tflite_ssd_graph.py \
	— pipeline_config_path=/Users/Alexandre/Dooble/training_demo/training/ssd_mobilenet_v2_apn.config \
	— trained_checkpoint_prefix=/Users/Alexandre/Dooble/training_demo/fine_tuned_model/model.ckpt \
	— output_directory=. \
	— add_postprocessing_op=true


#### Win10 Full script that works using the quantized model:
	conda activate tensorflow_115
	set PYTHONPATH=D:\TensorFlow\models\research;D:\TensorFlow\models\research\slim;%PYTHONPATH%
	python D:\TensorFlow\models\research\object_detection\export_tflite_ssd_graph.py --pipeline_config_path=D:\TensorFlow\private_project\trained_models\quantized_step_150931\ssd_mobilenet_v2_quantized_300x300_coco.config --trained_checkpoint_prefix=D:\TensorFlow\private_project\trained_models\quantized_step_150931\checkpoint\model.ckpt-150931 --output_directory=D:\TensorFlow\private_project\trained_models\quantized_step_150931\exported_ssd_pre_convert\ 
	--add_postprocessing_op=true
	
	tflite_convert --output_file="D:\TensorFlow\private_project\trained_models\quantized_step_150931\converted_tflite\model_quantized.tflite"  --graph_def_file="D:\TensorFlow\private_project\trained_models\quantized_step_150931\exported_ssd_pre_convert\tflite_graph.pb" --input_shape=1,300,300,3 --output_arrays=TFLite_Detection_PostProcess,TFLite_Detection_PostProcess:1,TFLite_Detection_PostProcess:2,TFLite_Detection_PostProcess:3 --input_arrays=normalized_input_image_tensor --allow_custom_ops

## MAC Full script  Changing max detection
    cd Dooble/trained_models/quantized_step_235247
	export PYTHONPATH=${PYTHONPATH}:${HOME}/TensorFlow/models-master/research/:${HOME}/TensorFlow/models-master/research/slim
	python ~/TensorFlow/models-master/research/object_detection/export_tflite_ssd_graph.py --pipeline_config_path=ssd_mobilenet_v2_quantized_300x300_coco.config --trained_checkpoint_prefix=checkpoint/model.ckpt-303591 --output_directory=exported_ssd_pre_convert --add_postprocessing_op=true --max_detections=20

	tflite_convert --output_file=converted_tflite/model_quantized.tflite  --graph_def_file=exported_ssd_pre_convert/tflite_graph.pb --input_shape=1,300,300,3 --output_arrays=TFLite_Detection_PostProcess,TFLite_Detection_PostProcess:1,TFLite_Detection_PostProcess:2,TFLite_Detection_PostProcess:3 --input_arrays=normalized_input_image_tensor --allow_custom_ops

## Nano
	export PYTHONPATH=${PYTHONPATH}:${HOME}/TensorFlow/models/research/:${HOME}/TensorFlow/models/research/slim

#### Changing 

It sounds that there is an option to setup:
https://stackoverflow.com/questions/58052869/tf-lite-object-detection-only-returning-10-detections
While invoking object_detection/export_tflite_ssd_graph.py, you would need to pass in the parameter --max_detections=20. Then, your change of NUM_DETECTIONS should work as expected.

##### Using a saved model

if saved_model.pb exist in saved_model folder 
Note: do not give the path to the file, but to the folder containing the file

	tflite_convert --output_file==D:\TensorFlow\private_project\convert_to_tflite\out.tflite --saved_model_dir=D:\TensorFlow\private_project\convert_to_tflite\input\saved_model\

	ValueError: None is only supported in the 1st dimension. Tensor 'image_tensor' has invalid shape '[None, None, None, 3]'.

Another attempt, using the pb file created with export_ssd_py
	tflite_convert --output_file==D:\TensorFlow\private_project\convert_to_tflite\out.tflite --saved_model_dir=D:\TensorFlow\private_project\export_to_tflite

Does not work, then i copy and renamed the orignal tflite_graph.pb to saved_model.pb

	RuntimeError: MetaGraphDef associated with tags {'serve'} could not be found in SavedModel. To inspect available tag-sets in the SavedModel, please use the SavedModel CLI: `saved_model_cli`

	tflite_convert --output_file=D:\TensorFlow\private_project\convert_to_tflite\out.tflite --graph_def_file=D:\TensorFlow\private_project\export_to_tflite\tflite_graph.pb

	tflite_convert --output_file=D:\TensorFlow\private_project\convert_to_tflite\out.tflite --graph_def_file=D:\TensorFlow\private_project\export_to_tflite\tflite_graph.pb --input_shape=1,300,300,3 --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' --input_arrays=normalized_input_image_tensor --allow_custom_ops


I had this error:

	2019-12-15 16:05:12.670476: F tensorflow/lite/toco/tooling_util.cc:935] Check failed: GetOpWithOutput(model, output_array) Specified output array "'TFLite_Detection_PostProcess'" is not produced by any op in this graph. Is it a typo? This should not happen. If you trigger this error please send a bug report (with code to reporduce this error), to the TensorFlow Lite team.

This was due to a bad output array: 

	tflite_convert --output_file=D:\TensorFlow\private_project\convert_to_tflite\out.tflite --graph_def_file=D:\TensorFlow\private_project\export_to_tflite\tflite_graph.pb --input_shape=1,300,300,3 --output_arrays="detection_scores","detection_boxes","detection_classes","detection_masks" --input_arrays=normalized_input_image_tensor --allow_custom_ops

This worked !!!

After looking on tensorboard, and also using graph info.py, on my windows setup, and the following created the tflite file.

	tflite_convert --output_file=D:\TensorFlow\private_project\convert_to_tflite\out.tflite --graph_def_file=D:\TensorFlow\private_project\export_to_tflite\tflite_graph.pb --input_shape=1,300,300,3 --output_arrays=TFLite_Detection_PostProcess --input_arrays=normalized_input_image_tensor --allow_custom_ops

But I was not able to use that file on my Xcode project : i got an error, so i used another network, a **quantized** one
D:\TensorFlow\private_project\training_demo\training\ssd_mobilenet_v2_quantized_300x300_coco.config

## Doing inference test
	python run_inference.py -g D:\TensorFlow\private_project\trained_models\quantized_step_150931\frozen_graph\frozen_inference_graph.pb -i D:\TensorFlow\private_project\dooble_pics\inf_test -l D:\TensorFlow\private_project\annotations\label_map.pbtxt

## Step 4a: Deploying the model on iPhone

Follow instruction for the Object Detection Tutorial.
Use the model and the label from from: Model_iOS folder.`

### Generate the icons for the application

Icons can be easily generate from the website:  
    https://appicon.co
Then from the zip file, replace only the AppIcon.appiconset folder

## Step 4b: Deploying the model on a Nvidia Jetson Nano

when running inference script : got error:
	tensorflow.python.framework.errors_impl.NotFoundError: Op type not registered 'TFLite_Detection_PostProcess' in binary running on nano. Make sure the Op and Kernel are registered in the binary running in this process. Note that if you are loading a saved graph which used ops from tf.contrib, accessing (e.g.) `tf.contrib.resampler` should be done before importing the graph, as contrib ops are lazily registered when the module is first accessed.tensorflow.python.framework.errors_impl.NotFoundError: Op type not registered 'TFLite_Detection_PostProcess' in binary running on nano. Make sure the Op and Kernel are registered in the binary running in this process. Note that if you are loading a saved graph which used ops from tf.contrib, accessing (e.g.) `tf.contrib.resampler` should be done before importing the graph, as contrib ops are lazily registered when the module is first accessed.

