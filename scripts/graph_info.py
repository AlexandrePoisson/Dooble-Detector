import tensorflow as tf
import argparse


def main():
    # Initiate argument parser
    parser = argparse.ArgumentParser(
        description="Graph Info")
    parser.add_argument("-i",
                        "--inputGraph",
                        help="Path to the pb",
                        type=str)
    gf = tf.GraphDef()
    args = parser.parse_args()
    m_file = open(args.inputGraph,'rb')
    gf.ParseFromString(m_file.read())

    with open('somefile.txt', 'a') as the_file:
        for n in gf.node:
            the_file.write(n.name+'\n')

    _file = open('somefile.txt','r')
    data = _file.readlines()
    print("Output name = ")
    print(data[len(data)-1])

    print("Input name = ")
    _file.seek (0)
    print(_file.readline())

if __name__=='__main__':
    main()
