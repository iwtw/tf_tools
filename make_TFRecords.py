import sys
import os
import tensorflow as tf
import argparse
import json
from scipy.misc import imread
def create_record(data_path_list,output_name):
    writer = tf.python_io.TFRecordWriter(output_name+".tfrecord")

    data_root = ""
    for v in data_path_list[0].split('/')[:-2]:
        data_root += v
        data_root += "/"
    
    labels = os.listdir( data_root )
    label_to_idx = {}
    for idx , v in enumerate(labels):
        label_to_idx[v] = idx
    
    label_idx_to_path = {}
    for label in label_to_idx.keys():
        label_idx = label_to_idx[label]
        label_idx_to_path[label_idx] = data_root + label
    f = open(output_name + "_label.json" , "w") 
    f.write(json.dumps(label_idx_to_path))
    f.close()

    for index, path in enumerate(data_path_list):
        label = path.split('/')[-2]
        with open(path, 'rb') as f:
            img_raw = f.read()

        #print index,img_raw
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[ label_to_idx[label]]) ),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                }
            )
        )
        writer.write(example.SerializeToString())
    writer.close()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("datalist",help="datalist_filename")
    parser.add_argument("output",help="output_filename (without dot suffix)")
    return parser.parse_args()
if __name__ == "__main__":
    args = parse_args()
    data_path_list = open( args.datalist , "r" ).read().split('\n')
    data_path_list.pop( len(data_path_list) - 1  )
    create_record( data_path_list , args.output )
    

