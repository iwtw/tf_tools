import numpy as np
import os
from scipy.misc import imsave
def save_training_images(path , images_list,it,epoch ):
    "images_list : containing elements of same shape in the format of [NHWC]"
    m = len( images_list )
    N , H , W ,C = images_list[0].shape
    rows = min( N , 5 )
    outputs = np.zeros((  H*rows,W*m,C))
    for i in range(  rows ):
        for j in range(  m ):
            outputs[  i*H : (i+1) * H , j * W : (j+1) * W ] = images_list[j][i]
    imsave( path + "/epoch{}_it{}.jpg".format(epoch,it) , outputs  )
    
def save_images( path_list , images_batch ):
    print(images_batch.shape)
    for i in range( images_batch.shape[0] ):
        
        temp_path = ""
        path_list_split = path_list[i].split('/')
        for j in range( len( path_list_split ) - 1  ):
            temp_path += path_list_split[ j ] + "/"
            if not  os.path.exists( temp_path ):
                os.mkdir( temp_path )

        imsave( path_list[i] , images_batch[i] )

