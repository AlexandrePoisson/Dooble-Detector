# Analyze the pictures contained in a folder given as argument
# _list all classes
# _give nb of occurences for each class
# _give a warning if there are less pictures than threshold

import os
import glob
import argparse
import xml.etree.ElementTree as ET


def xml_to_label_map(path):
    label_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object/name'):
            value = member.text
            label_list.append(value)
    return label_list



def main():
    # Initiate argument parser
    parser = argparse.ArgumentParser(
        description="Picture analyzer")
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
    main()
