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
	conda create --name tensorflow_env
	conda activate tensorflow_env
	conda install tensorflow=1.15 lxml pillow matplotlib
## Step 2: Training

### Transfer Learning


### Using colab


## Step 3: Transferring to Nano



## Step 4: Implementing on Nano

### Get the trained model
