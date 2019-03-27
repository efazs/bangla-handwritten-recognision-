# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 03:18:05 2019

@author: Efas
"""

# Import libraries
import os,cv2
import numpy as np

from PIL import Image

from numpy import *
import numpy as np



path1 =r"G:\pics\before_reform_originalS"#path containg the original image
#path1=r"G:\pics\pilo\before_reform_originals\0"
#path2 =r"G:\pics\after_reform" 
path2 =r"G:\pics\pilo\before_reform_originals" #path where the new image is to be stored
cvpath=path2



image_size=56#56
filter_number=32#64
Batch_size=10#64
dropoutvar=0.2# default for so many days 0.3
classsize=12
classwidth=250
#image_size=32#110
cv_imsize=(image_size,image_size)


listening = os.listdir(path1)
num_samples = size(listening)

#for file in listening:
 #   im = Image.open(path1+'\\'+file)
  #  img = im.resize((image_size,image_size))
   # gray = img.convert('L')
   # gray.save(path2+'\\'+file,"JPEG")
   
   
   
   
   
for file in listening:
    #im = Image.open(path1+'\\'+file)
    im = cv2.imread(path1+'\\'+file)
    
    #img = im.resize((image_size,image_size))
    #gray = img.convert('L')
    gray= cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    (thresh,bn)=cv2.threshold(gray,128,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    #bn.save(path2+'\\'+file,"JPEG")
    img=cv2.resize(bn,cv_imsize,interpolation = cv2.INTER_AREA)
    cv2.imwrite(cvpath+'\\'+file,img) 
    

size_64 = cv_imsize
#angle=45
list=[]
for f in os.listdir('.'):
    if f.endswith('.png'):
        i=Image.open(f)
        fn,fext = os.path.splitext(f)
        print(fn)
        dst_im = Image.new("RGB", (64,64), "white" )#bkgrd size
        im = i.convert('RGBA')
        #dst_im.paste( rot, (5, 5), rot ) image bkground (height,length), image frame a fixed na rakhle rotate korle rotated part kete jay
        rot = im.rotate( 3, expand=1 ).resize(size_64)
        dst_im.paste( rot, (0, 0), rot )
        dst_im.save('0/{}L3{}'.format(fn,fext))
       
        rot = im.rotate( -3, expand=1 ).resize(size_64)
        dst_im.paste( rot, (0, 0), rot ) # image specific frame a fixed rekhe ghurale size small hote thake
        dst_im.save('0/{}R3{}'.format(fn,fext))
        
        rot = im.rotate( 6, expand=1 ).resize(size_64)
        dst_im.paste( rot, (0, 0), rot )
        dst_im.save('0/{}L6{}'.format(fn,fext))
        
        rot = im.rotate( -6, expand=1 ).resize(size_64)
        dst_im.paste( rot, (0, 0), rot )
        dst_im.save('0/{}R6{}'.format(fn,fext))
        
        rot = im.rotate( 9, expand=1 ).resize(size_64)
        dst_im.paste( rot, (0, 0), rot )
        dst_im.save('0/{}L9{}'.format(fn,fext))
        
        rot = im.rotate( -9, expand=1 ).resize(size_64)
        dst_im.paste( rot, (0, 0), rot )
        dst_im.save('0/{}R9{}'.format(fn,fext))
        
        rot = im.rotate( 12, expand=1 ).resize(size_64)
        dst_im.paste( rot, (0, 0), rot )
        dst_im.save('0/{}L12{}'.format(fn,fext))
        
        rot = im.rotate( -12, expand=1 ).resize(size_64)
        dst_im.paste( rot, (0, 0), rot )
        dst_im.save('0/{}R12{}'.format(fn,fext))
        
        rot = im.rotate( 15, expand=1 ).resize(size_64)
        dst_im.paste( rot, (0, 0), rot )
        dst_im.save('0/{}L15{}'.format(fn,fext))
        
        rot = im.rotate( -15, expand=1 ).resize(size_64)
        dst_im.paste( rot, (0, 0), rot )
        dst_im.save('0/{}R15{}'.format(fn,fext))
        
        