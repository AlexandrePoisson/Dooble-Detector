from lxml import etree
import pickle
output_dict = pickle.load(open("/Users/Alexandre/Dooble/scripts/save.p", "rb" ))
import os
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

img_full_path = '/Users/Alexandre/Dooble/dooble_pics/inf_test/image3.jpg'

print(os.path.basename(img_full_path))
img_name = os.path.basename(img_full_path)

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
height = etree.SubElement(size, "height")
depth = etree.SubElement(size, "depth")
segmented = etree.SubElement(annotation, "segmented")
segmented.text = "0"

for lab in output_dict.items():
    label_object = etree.SubElement(annotation, "object")
    name = etree.SubElement(label_object, "name")
    name.text = str(lab['detection_classes'])

    bndbox = etree.SubElement(label_object, "name")
    xmin = etree.SubElement(bndbox, "xmin")
    xmin.text = lab['detection_boxes']
    ymin = etree.SubElement(bndbox, "ymin")
    xmax = etree.SubElement(bndbox, "xmax")
    ymax = etree.SubElement(bndbox, "ymax")   
print(etree.tostring(annotation))
