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

def create_label_file(output_dict, img_full_path):
	#img_full_path = '/Users/Alexandre/Dooble/dooble_pics/inf_test/image3.jpg'
	#label_file = '/Users/Alexandre/Dooble/annotations/label_map.pbtxt'


	im = Image.open(img_full_path)
	pic_width, pic_height = im.size

	print(os.path.basename(img_full_path))
	img_name = os.path.basename(img_full_path)
	img_xml = 	os.path.join(os.path.dirname(os.path.abspath(img_full_path)),os.path.splitext(img_name)[0]+'.xml')
	print(img_xml)
	annotation = etree.Element("annotation")
	folder = etree.SubElement(annotation, "folder")
	folder.text = img_name #fixme
	filename = etree.SubElement(annotation, "filename")
	filename.text = img_name

	source = etree.SubElement(annotation, "source")
	source.text = img_name
	database = etree.SubElement(source, "database")
	size = etree.SubElement(annotation, "source")
	width = etree.SubElement(size, "width")
	width.text = str(pic_width)
	height = etree.SubElement(size, "height")
	height.text = str(pic_height)
	depth = etree.SubElement(size, "depth")
	segmented = etree.SubElement(annotation, "segmented")
	segmented.text = "0"

	threshold = 0.4
	for i in range(0, output_dict['num_detections']):
		if output_dict['detection_scores'][i] > threshold:
			_item = output_dict['detection_boxes'][i]
			label_object = etree.SubElement(annotation, "object")
			name = etree.SubElement(label_object, "name")
			name.text = str(int_to_label(output_dict['detection_classes'][i]))

			# see here : https://stackoverflow.com/questions/48915003/get-the-bounding-box-coordinates-in-the-tensorflow-object-detection-api-tutorial
			bndbox = etree.SubElement(label_object, "bndbox")

			xmin = etree.SubElement(bndbox, "xmin")
			xmin.text = str(pic_width * output_dict['detection_boxes'][i][1])
			ymin = etree.SubElement(bndbox, "ymin")
			ymin.text = str(pic_height * output_dict['detection_boxes'][i][0])
			xmax = etree.SubElement(bndbox, "xmax")
			xmax.text = str(pic_width * output_dict['detection_boxes'][i][3])
			ymax = etree.SubElement(bndbox, "ymax")   
			ymax.text = str(pic_height * output_dict['detection_boxes'][i][2])

	f = open(img_xml, 'wb')
	f.write(etree.tostring(annotation, pretty_print=True))
	f.close()

if __name__=='__main__':
	output_dict = pickle.load(open("/Users/Alexandre/Dooble/scripts/save.p", "rb" ))
	img_full_path = '/Users/Alexandre/Dooble/dooble_pics/inf_test/image3.jpg'
	create_label_file(output_dict, img_full_path)
