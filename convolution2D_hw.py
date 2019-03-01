# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 18:18:57 2019

@author: h2r
"""

import cv2
import numpy as np
import math

#generate the gaussian kernal 
def gaussianKernal(k,size,sima):
    
    gau_total=0
    gau_f=np.zeros(shape=(size,size))
    bd=size//2
    for i in range(-bd, size-bd):
        for j in range(-bd, size-bd):
            pa=-(i*i+j*j)/(2*sima*sima)
            res=k*math.exp(pa)
            gau_f[i+bd,j+bd]=res
            gau_total=gau_total+res
    return gau_f,gau_total

#rotate the kernel for convolution
def rotate_kernel(kernel):
    rotated=kernel.copy()
    
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            rotated[i][j]=kernel[kernel.shape[0]-i-1][kernel.shape[1]-j-1]

    return rotated

#filter function for the convolution
#although the time capacity, I follow the pseudo-code of  
#https://en.wikipedia.org/wiki/Kernel_(image_processing) 
def sp_filter(f, w):
      
    k_rotate=rotate_kernel(w)  #call rotate the kernel function
    
    output_img=np.zeros(f.shape)
    image_h=f.shape[0]
    image_w=f.shape[1]
    
    kernel_h=k_rotate.shape[0]
    kernel_w=k_rotate.shape[1]
    
    h=kernel_h//2 
    w=kernel_w//2
    
    #convolution calculation
    for i in range(image_h):
        for j in range(image_w):
            sum=0
            
            for m in range(kernel_h):
                for n in range(kernel_w):
                    if 0<= i-h+m <image_h and 0<=j-w+n < image_w:
                        xx=i-h+m
                        yy=j-w+n
                        sum = sum + f[xx][yy]*k_rotate[m][n]
            output_img[i][j]=sum
    
    return output_img

def main():
    
    imgpath="4.1.05.tiff"
    img=cv2.imread(imgpath,cv2.IMREAD_GRAYSCALE) 
    cv2.imshow('org', img)
    
    k_g=input("K value for gaussian filter?") 
    size_g=input("size for gaussian filter?")
    sima_g=input("sima value for gaussian filter?")
    
    #gaussian kernal generate function
    gau_k, gau_total=gaussianKernal(int(k_g), int(size_g), int(sima_g))
    
    gau_final=gau_k/gau_total   #average the gaussian kernel 
    
    gau_output=sp_filter(img,  gau_final)
    gau_img=gau_output.astype(np.uint8)
    
    #show the img of the effect of gaussian filter
    cv2.imshow('gau_img',gau_img)
    
    #laplacian kernel    
    lapla_k=np.array(([0,-1,0],
                      [-1,4,-1],
                      [0,-1,0]), np.float32)
 
    #sharpen_output=sp_filter(img,  lapla_k)
    #sharpen_img=sharpen_output.astype(np.uint8)
    #cv2.imshow('sharpen_img',sharpen_img)
    #sharpen the image after blur the image
    output_log=sp_filter(gau_output,  lapla_k)
    
    c=input("c value for the final image?")
       
    #final image = origianl image + parameter * laplacian the gaussian image 
    final_out=img+float(c)*output_log
    
    #output the final image of the effect of 
    img_output = final_out.astype(np.uint8)       
    cv2.imshow('final',img_output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ =='__main__':
    main()