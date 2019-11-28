"""
Usage:

# Create train data:
python generate_tfrecord.py --label=<LABEL> --csv_input=<PATH_TO_ANNOTATIONS_FOLDER>/train_labels.csv  --output_path=<PATH_TO_ANNOTATIONS_FOLDER>/train.record

# Create test data:
python generate_tfrecord.py --label=<LABEL> --csv_input=<PATH_TO_ANNOTATIONS_FOLDER>/test_labels.csv  --output_path=<PATH_TO_ANNOTATIONS_FOLDER>/test.record
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf
import sys
sys.path.append("../../models/research")

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('label', '', 'Name of class label')
# if your image has more labels input them as
# flags.DEFINE_string('label0', '', 'Name of class[0] label')
# flags.DEFINE_string('label1', '', 'Name of class[1] label')
# and so on.
flags.DEFINE_string('img_path', '', 'Path to images')
FLAGS = flags.FLAGS


def class_text_to_int(row_label):
    if row_label == 'ampoule':
        return 1
    if row_label == 'ancre':
        return 2
    if row_label == 'araignee':
        return 3
    if row_label == 'arbre':
        return 4
    if row_label == 'biberon':
        return 5
    if row_label == 'bombe':
        return 6
    if row_label == 'bonhomme':
        return 7
    if row_label == 'bonhomme_neige':
        return 8
    if row_label == 'bouche':
        return 9
    if row_label == 'bougie':
        return 10
    if row_label == 'cactus':
        return 11
    if row_label == 'cadenas':
        return 12
    if row_label == 'carotte':
        return 13
    if row_label == 'chat':
        return 14
    if row_label == 'cheval':
        return 15
    if row_label == 'chien':
        return 16
    if row_label == 'cible':
        return 17
    if row_label == 'ciseau':
        return 18
    if row_label == 'cle':
        return 19
    if row_label == 'clown':
        return 20
    if row_label == 'coccinelle':
        return 21
    if row_label == 'coeur':
        return 22
    if row_label == 'crayon':
        return 23
    if row_label == 'dauphin':
        return 24
    if row_label == 'dinosore':
        return 25
    if row_label == 'dooble':
        return 26
    if row_label == 'dragon':
        return 27
    if row_label == 'eclair':
        return 28
    if row_label == 'fantome':
        return 29
    if row_label == 'feu':
        return 30
    if row_label == 'feuille':
        return 31
    if row_label == 'fleur':
        return 32
    if row_label == 'flocon':
        return 33
    if row_label == 'fromage':
        return 34
    if row_label == 'glacon':
        return 35
    if row_label == 'goutte':
        return 36
    if row_label == 'horloge':
        return 37
    if row_label == 'igloo':
        return 38
    if row_label == 'lune':
        return 39
    if row_label == 'lunette':
        return 40
    if row_label == 'marteau':
        return 41
    if row_label == 'note':
        return 42
    if row_label == 'oeuil':
        return 43
    if row_label == 'oiseau':
        return 44
    if row_label == 'point_exclamation':
        return 45
    if row_label == 'point_interrogation':
        return 46
    if row_label == 'pomme':
        return 47
    if row_label == 'sens_interdit':
        return 48
    if row_label == 'soleil':
        return 49
    if row_label == 'tache':
        return 50
    if row_label == 'tete_mort':
        return 51
    if row_label == 'toile':
        return 52
    if row_label == 'tortue':
        return 53
    if row_label == 'trefle':
        return 54
    if row_label == 'voiture':
        return 55
    if row_label == 'ying_yang':
        return 56
    if row_label == 'zebre':
        return 57
    else:
        return 0


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with open(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    # check if the image format is matching with your images.
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    writer = tf.io.TFRecordWriter(FLAGS.output_path)
    path = os.path.join(os.getcwd(), FLAGS.img_path)
    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.app.run()