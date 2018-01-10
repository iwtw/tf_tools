import sys
import time 
import os
import tensorflow as tf
import argparse
import json
from scipy.misc import imread
def create_record(data_path_list,output_name):
    writer = tf.python_io.TFRecordWriter(output_name+'.tfrecord')

    data_root = ''
    for v in data_path_list[0].split('/')[:-2]:
        data_root += v
        data_root += '/'
    
    if False:
        labels = os.listdir( data_root )
        label_to_idx = {}
        for idx , v in enumerate(labels):
            label_to_idx[v] = idx
        
        label_idx_to_path = {}
        for label in label_to_idx.keys():
            label_idx = label_to_idx[label]
            label_idx_to_path[label_idx] = data_root + label
        f = open(output_name + '_label.json' , 'w') 
        f.write(json.dumps(label_idx_to_path))
        f.close()

    pre_time = time.time()
    for index, lr_path in enumerate(data_path_list):

        lr_path_split = lr_path.split('/')
        hr_path = ""
        for j in range( len(lr_path_split) - 3  ):
            hr_path += lr_path_split[j]
            hr_path += "/"
        dir_name = lr_path_split[-3]
        dir_name_split = dir_name.split('_')
        for k in range( len( dir_name_split) - 1 ):
            if k!=0:
                hr_path+= "_"
            hr_path += dir_name_split[k]
        hr_path += '/'
        hr_path += lr_path_split[-2] + '/'
        hr_path += lr_path_split[-1]

        with open(lr_path, 'rb') as f:
            image_LR = f.read()
        with open(hr_path , 'rb' ) as f:
            image_HR = f.read()

        #print index,img_raw
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'image_LR': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_LR])),
                    'image_HR': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_HR]))
                }
            )
        )
        writer.write(example.SerializeToString())
        if index%1024==1024 - 1 :
            temp_time = time.time() 
            print( str(index + 1) +" examples done , speed: %.1f/s"%(1024/(temp_time - pre_time))  )
            pre_time = temp_time
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
    

