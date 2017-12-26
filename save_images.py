import numpy as np
import os
from scipy.misc import imsave
def save_training_images( images_list,epoch, output_dir ="training_)output/" ):
    "images_list : containing elements of same shape in the format of [NHWC]"
    m = len( images_list )
    for i in range(m):
        images_list[i] = np.array( images_list[i] )
    N , H , W ,C = images_list[0].shape
    rows = min( N , 10 )
    outputs = np.zeros((  H*rows,W*m,C))
    for i in range(  rows ):
        for j in range(  m ):
            outputs[  i*H : (i+1) * H , j * W : (j+1) * W ] = images_list[j][i]
    imsave( output_dir + "/epoch%2d.jpg"%(epoch) , outputs  )
    
def save_images(  filename_list , images , output_dir = "output/" ):
    #假设filename_list是输入的list，且以/label/image.png的形式存储
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if output_dir[-1] !="/":
        output_dir+="/"
   # print(images_batch.shape)
    for i in range( images.shape[0] ):
        #对第i个图像
        
        #若作为label的目录不存在，则创建目录
        temp_dir =  output_dir
        filename_list_split = filename_list[i].split("/")
        temp_dir += filename_list_split[ -2 ] + "/"
        if not  os.path.exists( temp_dir ):
            os.mkdir( temp_dir )
        imsave( temp_dir + filename_list_split[-1] , images[i]  )

