# Batch image resizer
from PIL import Image
import argparse
import os

def resizer(input_path, output_path, start_index=123):
    print("Resizer")

    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(input_path):
        for file in f:
            print(file)
            if '.jpeg' in file.lower():
                files.append(os.path.join(r, file))
            elif '.jpg' in file.lower():
                files.append(os.path.join(r, file))

    basewidth = 600

    for f in files:
        print(f)
        img = Image.open(f)
        wpercent = (basewidth/float(img.size[0]))
        hsize = int((float(img.size[1])*float(wpercent)))
        img = img.resize((basewidth,hsize), Image.ANTIALIAS)
        output_path_f = output_path + '\\dooble{}.jpg'.format(start_index)
        print(output_path_f)
        img.save(output_path_f)
        start_index = start_index + 1


def main():
    print("Start")
    # Initiate argument parser
    parser = argparse.ArgumentParser(
        description="Batch Image resizer and renamer")
    parser.add_argument("-i",
                        "--inputDir",
                        help="Path to the folder where the input .jpg files are stored",
                        type=str)
    parser.add_argument("-o",
                        "--outputDir",
                        help="Path to the folder where the output .jpg files will be stored", 
                        type=str)
    args = parser.parse_args()

    if(args.inputDir is None):
        args.inputDir = os.getcwd()
    if(args.outputDir is None):
        args.outputDir = "D:\\TensorFlow\\private_project\\dooble_pics\\TO_LABEL"

    assert(os.path.isdir(args.inputDir))

    resizer(args.inputDir, args.outputDir)

if __name__ == '__main__':
    main()