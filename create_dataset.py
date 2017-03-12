# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 11:00:35 2017

@author: vijay
"""

import os
import cv2
from shutil import copyfile
from random import randint,shuffle

###############################################################################

def dir_check(location):
    
    if not os.path.exists(location):
        os.makedirs(location)

###############################################################################
def main():
    
    root='/media/vijay/Vijay/Datasets/NIST/digits/'
    dst='dataset/'

    dir_check(dst+'train/')
    dir_check(dst+'val/')
    
    
    classes=os.listdir(root)
    
    for clas in classes:
        
        imlist=os.listdir(root+clas)
        shuffle(imlist)
        
        train=imlist[:2000]
        val=imlist[2001:4001]
        
        ## write train and val files
        
        dir_check(dst+'train/'+clas)
        dir_check(dst+'val/'+clas)
        
        for tmp in train:
            
            fname=str(randint(0,999999999999))
            img=cv2.imread(root+clas+'/'+tmp)
            cv2.imwrite(dst+'train/'+clas+'/'+fname+'.jpg',img)
        
        for tmp in val:
            
            fname=str(randint(0,999999999999))
            img=cv2.imread(root+clas+'/'+tmp)
            cv2.imwrite(dst+'val/'+clas+'/'+fname+'.jpg',img)
    

if __name__=='__main__':
    main()

