import tensorflow as tf
import argparse
import data_input
import save_images
import os 

def argparse():
    parser = argparse.ArgumentParser( description = "bicubic")
    parser.add_argument("-inputpath",help="input tfrecord path")
    parser.add_argument("-height",type=int)
    parser.add_argument("-width",type=int)
    parser.add_argument("-scale",type=int , help ="'scale < 0' means fraction ")
    parser.add_argument("--batch_size",type=int , default = 64)
    parser.add_argument("--n_gpus",type=int , default = 1)
    return parser.argparse()
    
def init():
    if !os.path.exists("outputs"):
        os.mkdir("outputs")
        

def main(_):
    init()
    args = argparse()
    file_queue = tf.train.string_input_producer([args.inputpath])
    inputs = data_input.get_batch( file_queue , [args.height,args.width] ,  64 , 4 ,5 , is_training = false )
    output_height = args.height * args.scale
    output_width = args.width * args.scale
    if ( args.scale < 0 ):
        assert (args.height % args.scale == 0 )
        assert ( args.width % args.scale == 0 )
        output_height = args.height / args.scale
        output_width = args.width / args.width

    outputs = tf.image.resize_bicubic(inputs , [output_height , output_width ])
    if
    save_images


    for device_index in range(args.gpus)
        with tf.device("gpus:/{}".format(device_index))  :



if __name__ == "__main__":
    tf.app.run(main)

