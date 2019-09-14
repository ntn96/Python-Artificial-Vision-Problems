#!/usr/bin/env python

import cv2 as cv                                                    #Antonio Serrano Miralles
import numpy as np
import skimage.io  as io

def rgb2gray(x):
    return cv.cvtColor(x,cv.COLOR_RGB2GRAY)

def readrgb(name):
    return cv.imread(name)

def t(h,x):
    return cv.warpPerspective(x, desp((100,150)) @ h,(1000,600))

#aplica un desplazamiento
def desp(d):
    dx,dy = d
    return np.array([
            [1,0,dx],
            [0,1,dy],
            [0,0,1]])

def fuse(sift,bf,gray1,gray2,rgb1,rgb2):
    (kps1, descs1) = sift.detectAndCompute(gray1, None) #Calculamos los keypoints de cada una de las imágenes
    (kps2, descs2) = sift.detectAndCompute(gray2, None)
    matches1_2 = bf.knnMatch(descs2,descs1,k=2)         #Calcula las coincidencias de keypoints entre las dos imágenes
    good1_2 = []
    for m,n in matches1_2:                              #Filtramos los keypoints ambiguos para quedarnos con los más veraces
        if m.distance < 0.75*n.distance:
            good1_2.append(m) 
    #Llenamos los arrays con los keypoints origen y destino para poder pasarselos correctamente a findHomography
    src_pts1_2 = np.array([ kps1 [m.trainIdx].pt for m in good1_2 ]).astype(np.float32).reshape(-1,2)
    dst_pts1_2 = np.array([ kps2 [m.queryIdx].pt for m in good1_2 ]).astype(np.float32).reshape(-1,2)
    H1_2, mask1_2 = cv.findHomography(src_pts1_2, dst_pts1_2, cv.RANSAC, 3) #Busca la transformación proyectiva para fusionar correctamente
    return np.maximum(t(np.eye(3),rgb2), t(H1_2,rgb1))  #Retornamos la imagen fusionada en color

img1  = readrgb('pano001.jpg')
img2  = readrgb('pano002.jpg')
img3  = readrgb('pano003.jpg')
gray1 = rgb2gray(img1)
gray2 = rgb2gray(img2)
gray3 = rgb2gray(img3)

sift = cv.xfeatures2d.SIFT_create()
bf = cv.BFMatcher()
aux = fuse(sift,bf,gray1,gray2,img1,img2)   #Para n imágenes hay que realizar n-1 fusiones
aux = fuse(sift,bf,aux,gray3,aux,img3)

cv.namedWindow("resultado")
cv.imshow('resultado',aux);
while(True):
    k = cv.waitKey(1) & 0xFF
    if k == 27: break

cv.destroyAllWindows() 
