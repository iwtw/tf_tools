import numpy as np
from scipy.misc import imsave
def save_images(path,images_list,it,epoch):
    "images_list : containing elements of same shape in the format of [NCHW]"
    if not path[-1] == "/":
        path += "/"
    m = len( images_list )

    for i in range(m ):
        images_list[i] = np.transpose(images_list[i] ,[0,3,1,2] )
    N , H , W ,C = images_list[0].shape
    rows = min( N , 5 )
    outputs = ((  H*rows,W*m,C))
    for i in range(  rows ):
        for j in range(  m ):
            outputs[  i*H : (i+1) * H , j * W : (j+1) * W ] = images_list[j][i]
    imsave( path + "epoch{}_it{}".format(it,epoch) , outputs  )
    
    
