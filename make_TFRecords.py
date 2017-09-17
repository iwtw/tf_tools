import sys
import os
import tensorflow as tf
import argparse
from scipy.misc import imread
def create_record(data_path_list,output_path):
    writer = tf.python_io.TFRecordWriter(output_path)

    data_root = ""
    for v in data_path_list[0].split('/')[:-2]:
        data_root += v
        data_root += "/"
    
    labels = os.listdir( data_root )
    idx_labels = {}
    for idx , v in enumerate(labels):
        idx_labels[v] = idx
    

    for index, path in enumerate(data_path_list):
        label = path.split('/')[-2]
        img = imread(  path )
       # img = img.resize((28, 28))
        img_raw = img.tobytes() 
        #print index,img_raw
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[ idx_labels[label]]) ),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                }
            )
        )
        writer.write(example.SerializeToString())
    writer.close()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("datalist",help="datalist_filename")
    parser.add_argument("output",help="output_filename")
    return parser.parse_args()
if __name__ == "__main__":
    args = parse_args()
    data_path_list = open( args.datalist , "r" ).read().split('\n')
    data_path_list.pop( len(data_path_list) - 1  )
    create_record( data_path_list , args.output )
    

