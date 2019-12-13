from lxml import etree
import pickle
import os
from PIL import Image
from int_to_label import int_to_label
"""
<annotation>
	<folder>TO_LABEL</folder>
	<filename>dooble33.jpg</filename>
	<path>/Users/Alexandre/Dooble/dooble_pics/TO_LABEL/dooble33.jpg</path>
	<source>
		<database>Unknown</database>
	</source>
	<size>
		<width>640</width>
		<height>480</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
	<object>
		<name>tete_mort</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>68</xmin>
			<ymin>305</ymin>
			<xmax>144</xmax>
			<ymax>365</ymax>
		</bndbox>
	</object>
</annotation>
"""

def create_label_file(output_dict, img_full_path, threshold = 0.7):
	#img_full_path = '/Users/Alexandre/Dooble/dooble_pics/inf_test/image3.jpg'
	#label_file = '/Users/Alexandre/Dooble/annotations/label_map.pbtxt'


	im = Image.open(img_full_path)
	pic_width, pic_height = im.size
	print("Full path: {}".format(img_full_path))
	folder = os.path.dirname(img_full_path)
	img_name = os.path.basename(img_full_path)
	img_xml = os.path.join(os.path.dirname(os.path.abspath(img_full_path)),os.path.splitext(img_name)[0]+'.xml')
	annotation = etree.Element("annotation")
	folder = etree.SubElement(annotation, "folder")
	folder.text =  os.path.dirname(img_full_path)
	filename = etree.SubElement(annotation, "filename")
	filename.text = img_name

	path = etree.SubElement(annotation, "path")
	path.text = img_full_path

	source = etree.SubElement(annotation, "source")

	database = etree.SubElement(source, "database")
	database.text = "Unknown"

	size = etree.SubElement(annotation, "size")
	width = etree.SubElement(size, "width")
	width.text = str(pic_width)
	height = etree.SubElement(size, "height")
	height.text = str(pic_height)
	depth = etree.SubElement(size, "depth")
	depth.text = "3"
	segmented = etree.SubElement(annotation, "segmented")
	segmented.text = "0"

	
	for i in range(0, output_dict['num_detections']):
		if output_dict['detection_scores'][i] > threshold:
			_item = output_dict['detection_boxes'][i]
			label_object = etree.SubElement(annotation, "object")
			name = etree.SubElement(label_object, "name")
			name.text = str(int_to_label(output_dict['detection_classes'][i]))
			pose = etree.SubElement(label_object, "pose")
			pose.text = "Unspecified"
			truncated = etree.SubElement(label_object, "truncated")
			truncated.text = "0"			
			difficult = etree.SubElement(label_object, "difficult")
			difficult.text = "0"	
			
			# see here : https://stackoverflow.com/questions/48915003/get-the-bounding-box-coordinates-in-the-tensorflow-object-detection-api-tutorial
			bndbox = etree.SubElement(label_object, "bndbox")

			xmin = etree.SubElement(bndbox, "xmin")
			xmin.text = str(int(round(pic_width * output_dict['detection_boxes'][i][1])))
			ymin = etree.SubElement(bndbox, "ymin")
			ymin.text = str(int(round(pic_height * output_dict['detection_boxes'][i][0])))
			xmax = etree.SubElement(bndbox, "xmax")
			xmax.text = str(int(round(pic_width * output_dict['detection_boxes'][i][3])))
			ymax = etree.SubElement(bndbox, "ymax")   
			ymax.text = str(int(round(pic_height * output_dict['detection_boxes'][i][2])))

	f = open(img_xml, 'wb')
	f.write(etree.tostring(annotation, pretty_print=True))
	f.close()

def main(output_dict, img_full_path, threshold):
	create_label_file(output_dict, img_full_path, threshold)

if __name__=='__main__':
	output_dict = pickle.load(open("/Users/Alexandre/Dooble/scripts/save.p", "rb" ))
	img_full_path = '/Users/Alexandre/Dooble/dooble_pics/inf_test/image3.jpg'
	threshold = 0.7
	main(output_dict,create_label_file, threshold)

