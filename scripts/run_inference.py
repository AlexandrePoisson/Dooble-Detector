
"""
Script that is running inference, based on a set of images or a video stream
Example of usage:
  python run_inference.py -g D:\TensorFlow\private_project\convert_to_tflite\input\frozen_inference_graph.pb -i D:\TensorFlow\private_project\dooble_pics\inf_test -l D:\TensorFlow\private_project\annotations\label_map.pbtxt
"""

"""
# Section below is to be reworked. It is currently disabled, the user is expected to have 
path_to_model = r'D:\TensorFlow\models'
# path_to_model = r'/Users/Alexandre/TensorFlow/models-master'
var_env = ':' + path_to_model + '/research/:' + path_to_model + '/research/slim'
print(var_env)
os.environ['PYTHONPATH'] += var_env
sys.path.append(os.path.join(path_to_model,'research'))
sys.path.append(os.path.join(path_to_model,'utils'))
"""

"""
#Note : this script  needs the protos file to be created
(base) D:\TensorFlow\models\research>conda activate tensorflow_115
(tensorflow_115) D:\TensorFlow\models\research>protoc object_detection/protos/*.proto --python_out=.
"""
import argparse
import numpy as np
import os
import sys
import tensorflow as tf
import matplotlib
from matplotlib import pyplot as plt

from PIL import Image
import glob, os

from generate_labelimg_annotation_xml import create_label_file
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
matplotlib.use("TkAgg")
# May meet problem when loading a new pb file
tf.contrib.resampler

def run_inference_video_stream(detection_graph, category_index):
  import cv2
  cap = cv2.VideoCapture(0)
  while True:
    ret, image_np = cap.read()
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np, detection_graph)
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks'),
      use_normalized_coordinates=True,
      line_thickness=8)
    cv2.imshow('object detection', cv2.resize(image_np, (800,600)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break


def run_inference_images(PATH_TO_TEST_IMAGES_DIR, detection_graph, category_index):
  # If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
  
  for image_path in glob.glob("{}/*.jpg".format(PATH_TO_TEST_IMAGES_DIR)):
    print("Image path in run_inference.py: {}".format(image_path))
    image = Image.open(image_path)
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np, detection_graph)
    # Visualization of the results of a detection.

    # Create a labelImg xml file
    create_label_file(output_dict, os.path.abspath(image_path))

    # Display Inference result
    plt.show(block=False)
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=8)
    plt.figure()
    plt.imshow(image_np)
  plt.show()

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict


def main():
  print("Start")
  # Initiate argument parser
  parser = argparse.ArgumentParser(
      description="Batch Image resizer and renamer")
  parser.add_argument("-g",
                      "--inputGraph",
                      help="Path to the frozen graph to be used: eg.: graph/frozen_inference_graph.pb",
                      type=str)

  parser.add_argument("-i",
                      "--inputFolder",
                      help="PATH_TO_TEST_IMAGES_DIR",
                      type=str)
  parser.add_argument("-l",
                      "--inputLabel",
                      help="Path to the label file: eg: annotations/label_map.pbtxt", 
                      type=str)
  parser.add_argument("-s", "--inputSource", help="input source: video or image", type=str)
  args = parser.parse_args()

  if(args.inputGraph is None):
    raise AttributeError("inputGraph")
  if(args.inputFolder is None and args.inputSource is "image"):
    raise AttributeError("inputFolder")
  if(args.inputLabel is None):
    raise AttributeError("inputLabel")
  if(args.inputSource is None):
    args.inputSource="image"


  # What model to download.
  # Path to frozen detection graph. This is the actual model that is used for the object detection.
  PATH_TO_CKPT = args.inputGraph
  # List of the strings that is used to add correct label for each box.

  PATH_TO_LABELS = args.inputLabel
  NUM_CLASSES = 57

  detection_graph = tf.Graph()
  with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')
      

  label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
  categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
  category_index = label_map_util.create_category_index(categories)
  
  input_source = args.inputSource

  if input_source == "image":
    PATH_TO_TEST_IMAGES_DIR = args.inputFolder
    run_inference_images(PATH_TO_TEST_IMAGES_DIR, detection_graph, category_index)
  elif input_source == "video":
    run_inference_video_stream(detection_graph, category_index)
  else:
    print("bad argument")

  

if __name__ == '__main__':
    main()

