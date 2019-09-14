#!/usr/bin/env python

import cv2         as cv                           #Antonio Serrano Miralles
import numpy       as np

cap = cv.VideoCapture(0)                           #Obtengo la camara
global estaticoActual                              #Guarda la última imagen en la que se detecto movimiento
global cambio                                      #Se pone a true cuando se ha detectado movimiento en el frame actual
cambio = False
ret, estaticoActual = cap.read()     
sensibilidad = 30                                  #Determina cuan diferente debe ser la imagen para que se considere que se ha detectado 
                                                   #movimiento, cuanto mayor sea el número más tolerante es en cuanto a cambios

def bgr2yuv(x):					
    return cv.cvtColor(x,cv.COLOR_BGR2YUV)

def generator(frame):
    global estaticoActual
    global cambio
    ea_yuv = bgr2yuv(estaticoActual)               #Paso a yuv las dos imagenes
    frame_yuv = bgr2yuv(frame)
    ea_uv = ea_yuv.astype(float)[:,:,[1,2]]        #Me libro del canal de luminosidad para no tenerlo en cuenta en las comparaciones
    frame_uv = frame_yuv.astype(float)[:,:,[1,2]]
    dif = abs(ea_uv-frame_uv)                      #Calculo la diferencia entre las imagenes
    if np.any(dif > sensibilidad):                 #Si algún pixel ha cambiado más del valor determinado por la sensibilidad
        estaticoActual = frame                     #Cambia el frame estatico actual
        cambio = True	

i = 0                                              #Variable que cuenta el número de veces que se ha detectado movimiento
while(cv.waitKey(1) & 0xFF != 27):                 #Mientras no se pulsa escape
    ret, frame = cap.read()                        #Lee frame de la cámara
    generator(frame)
    if cambio :                                    #Si se ha producido un cambio
        cambio = False                             #Resetea la variable cambio
        i = i + 1 
        print("Movimiento detectado", i)           #Imprime por consola que se ha detectado movimiento
        cv.imshow('estatico',frame)                #Modifica la ventana con el último frame estático
        cv.imshow('cambio', frame)                 #Actualiza la ventana que muestra video si detectar movimiento
    else :
        cv.imshow('cambio',frame)

cv.destroyAllWindows()
