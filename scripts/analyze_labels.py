# Analyze the labels contained in a folder given as argument


import os
import glob
import argparse
import xml.etree.ElementTree as ET
from PIL import Image

def xml_to_pics(path):
    label_list = []
    i = 0
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            i = i + 1
            value = member.find('name').text
            box = member.find(r'bndbox')
            x_min = box.find('xmin').text
            x_max = box.find('xmax').text
            y_min = box.find('ymin').text
            y_max = box.find('ymax').text
            im = Image.open(xml_file[:-3] + 'jpg')        
            im1 = im.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
            im1.save('temp\{}_{}.jpg'.format(value,os.path.basename(xml_file)))

    return label_list

def test():
    label_list = xml_to_pics(r'D:\TensorFlow\private_project\dooble_pics\train')

def main():
    # Initiate argument parser
    parser = argparse.ArgumentParser(
        description="Labels analyzer. Create pictures which names are <labels_name>_<picture_name>, so that user can review")
    parser.add_argument("-i",
                        "--inputDir",
                        help="Path to the folder where the input .xml files are stored",
                        type=str)
    args = parser.parse_args()

    if(args.inputDir is None):
        args.inputDir = os.getcwd()

    assert(os.path.isdir(args.inputDir))

    label_list = xml_to_label_map(args.inputDir)
    low_number_of_item_for_class = []
    nb_item_per_class_threshold = 10
    print("Number of Classes: {}".format(len(set(label_list))))

    print("Classes: {}".format(set(label_list)))
    for item in sorted(set(label_list)):
        try:
            print("item {} : # {}".format(item, label_list.count(item)))
        except UnicodeEncodeError as err:
            print("item {} : # {}".format(item.encode('utf-8'),label_list.count(item)))
            print(err)
        if label_list.count(item) < nb_item_per_class_threshold:
            low_number_of_item_for_class.append([item, label_list.count(item)])
    for item, nb in low_number_of_item_for_class:
        print("Warning: few element for {}: only {}".format(item, nb))


if __name__ == '__main__':
    test()
