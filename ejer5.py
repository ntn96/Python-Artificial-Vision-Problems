#!/usr/bin/env python

import cv2         as cv                          #Antonio Serrano Miralles
import numpy       as np
import skimage.io  as io

global background                                 #Frame que guarda el fondo detectado
global backSet                                    #Está a true cuando se ha colocado el frame de backgroud
global img
backSet = False

def setBackground(frame):
 global background
 global backSet
 background = frame
 backSet = True

def chroma(frame):
 global background
 global img
 back_bgr = background.astype(float)
 frame_bgr = frame.astype(float) 
 dif = np.sum(abs(frame_bgr-back_bgr),axis=2)      #Calculo la diferencia entre las imagenes fondo y la actual por los canales de color
 mask = dif > 75                                   #Calculo una máscara que tiene solo encuenta diferencias absolutas mayores de 75
 r,c = mask.shape
 res = cv.resize(img,(c,r))                        #Adecuamos el tamaño de la mascara al de la imagen sobre la que vamos a pegar los objetos
 mask3 = mask.reshape(r,c,1)                       #Convierto la máscara a 3 canales
 np.copyto(res,frame,where=mask3)                  #Pego los objetos en la imagen teniendo según la máscara
 cv.imshow('original', res)

cap = cv.VideoCapture(0)
img = io.imread('./pano001.jpg')                   #Carga una imagen con ese nombre que se encuentre en el mismo directorio
while(True):        
 ret, frame = cap.read()
 if not backSet:                                   #Si no se ha puesto el fondo
  cv.imshow('original', frame)                     #Imprimimos por la ventana original para que se vea en todo momento la imagen
 if backSet:                                       #Si hay fondo accedemos a la función chroma
  chroma(frame)
 k = cv.waitKey(1) & 0xFF
 if k== 27:                                        #Esc para salir
  break
 if k== ord(' '):                                  #Botón espacio para poner el fondo
  setBackground(frame)

cv.destroyAllWindows()
