import tensorflow as tf
import numpy as np
import argparse
from save_images import *

args = None
def _parse_function( filename ):
    image_string = tf.read_file( filename )
    image_decoded = tf.image.decode_image( image_string )
    image_decoded.set_shape([None,None,3])
    #print( tf.shape(image_decoded) )
    #image_resized = tf.image.resize_images( image_decoded , [28,24] )
    return image_decoded 

def build_graph():
    filenames = tf.placeholder( tf.string , shape = [None] )
    dataset = tf.data.TextLineDataset( filenames )
    dataset = dataset.map( _parse_function )
   # dataset = datase.repeat()
    dataset = dataset.batch(64)
    iterator = dataset.make_initializable_iterator()
    
    x = iterator.get_next()

    return filenames , iterator , x 


def test( filenames , iterator ,  x ):
    filename_list = open( args.input ).read().split('\n')
    if filename_list[-1]=="":
        filename_list.pop()
    
    print(filename_list[:10])
    print("-=-=-=-----")
    cnt_batch = 0 
    with tf.Session() as sess:
        sess.run( iterator.initializer , feed_dict = { filenames : [ args.input ]})
        while True:
            try:
                out = sess.run( x )
                save_images(filename_list[cnt_batch * args.batch_size : (cnt_batch+1) * args.batch_size ] , out, output_dir = "dataset_io" )
                cnt_batch += 1 
            except tf.errors.OutOfRangeError:
                break

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument( "-input" , "-i" )
    parser.add_argument( "-output" , "-o" )
    parser.add_argument( "-batch_size" , type=int , default = 64 )
    return parser.parse_args()

if __name__=="__main__":
    args = parse_args()
    if args.output[-1]!="/" :
        args.output+="/"
    filenames , iterator , x  = build_graph()
    test( filenames , iterator , x )
