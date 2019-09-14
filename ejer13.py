#!/usr/bin/env python

import cv2 as cv                                                    #Antonio Serrano Miralles
import numpy as np
import skimage.io  as io
import math

global pointsSelected
global ptos
global rec

def readrgb(name):
    return cv.imread(name)

# crea cómodamente un vector (array 1D)
def vec(*argn):
    return np.array(argn)

#Pasa un vector a coordenadas homogéneas
def homog(x):
    ax = np.array(x)
    uc = np.ones(ax.shape[:-1]+(1,))
    return np.append(ax,uc,axis=-1)

#Pasa un vector a coordenadas tradicionales 
def inhomog(x):
    ax = np.array(x)
    return ax[..., :-1] / ax[...,[-1]]

# aplica una transformación homogénea h a un conjunto
# de puntos ordinarios, almacenados como filas 
def htrans(h,x):
    return inhomog(homog(x) @ h.T)

#aplica un desplazamiento
def desp(d):
    dx,dy = d
    return np.array([
            [1,0,dx],
            [0,1,dy],
            [0,0,1]])

#aplica un reescalado
def scale(s):
    sx,sy = s
    return np.array([
            [sx,0,0],
            [0,sy,0],
            [0,0,1]])

def fun(event, x, y, flags, param):
    global pointsSelected
    global ptos 
    if event == cv.EVENT_LBUTTONDOWN :                          #Cuando detectamos un clic
        cv.circle(rec,(x,y),1,(255,0,0),2)                      #Dibujamos un punto en la posición del clic
        cv.imshow('imagen rectificada',rec)
        ptos.append((x,y))                                      #Añadimos el punto a la lista de puntos seleccinados
        pointsSelected+=1                                       #Actualizamos el valor de los puntos seleccionados
        if pointsSelected == 2 :                                #Si tenemos dos puntos elegidos
            xv = ptos[1][0] - ptos[0][0]                        #Calculamos las coordenadas X e Y del vector que los unen
            yv = ptos[1][1] - ptos[0][1]
            mod = math.sqrt(xv**2 + yv**2)                      #Hayamos el módulo del vector, es decir, la distancia entre ambos puntos
            print("Distancia real entre puntos",mod/20)
            ptos.clear()                                        #Vacio la lista y pongo a 0 el número de puntos seleccionados
            pointsSelected = 0                                  #Para poder hacer más de una medición

pointsSelected = 0                                      #Cantidad de puntos elegidos, cuando llevamos 2 se calcula la distancia
ptos = []                                               #Puntos seleccionados por el usuario
carnet = np.array([[0,0],[8.5,0],[0,5.5],[8.5,5.5]])    #Dimesiones del carnet en cm 
dst = htrans(desp(vec(100,80)) @ scale((20,20)), carnet)#Se aplica un desplazamiento y escalado para que la imagen pueda verse bien
puntos = np.array([[508,558],[626,312],[324,463],[467,258]])    #Posiciones de las esquinas del carnet
H,_= cv.findHomography(puntos, dst)                             #Busca la transformación proyectiva
cv.namedWindow("imagen rectificada")
cv.namedWindow("imagen original")
cv.setMouseCallback("imagen rectificada", fun)          #Pongo que los eventos de raton para la ventana webcam sean tratados por la función fun
img = readrgb('coins.png')
rec = cv.warpPerspective(img,H,(800,500))               #Aplico la rectificación del plano de la imagen y la imprimo en otra ventana
cv.imshow('imagen rectificada',rec)
cv.imshow('imagen original', img)
while(True):
    k = cv.waitKey(1) & 0xFF
    if k == 27: break

cv.destroyAllWindows() 
