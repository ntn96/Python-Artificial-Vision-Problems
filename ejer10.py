#!/usr/bin/env python

import cv2 as cv                                                    #Antonio Serrano Miralles
import numpy as np
import matplotlib.pyplot as plt
import skimage.io        as io

from matplotlib.pyplot   import imshow, subplot, title, plot

global modelosKP
global modelosIM
global bf
global sensibilidad

def capturaModelo(frame):
    modelosKP.append(method.detectAndCompute(frame,None)[1])
    modelosIM.append(frame)                                       #Guardamos la imagen en la lista de imágenes de modelo
    print("Modelo guardado, total: ",len(modelosIM))

def comparar(frame):
    imagenKP = method.detectAndCompute(frame,None)[1]
    matches = [bf.knnMatch(imagenKP, x, 2) for x in modelosKP]
    good = []
    mejorCoincidencia = 0
    mejorIndex = 0
    for i in range(len(matches)):                                   #Para los matches de cada comparacion
        good.append(0)
        tam = 0
        for best, second in matches[i]:
            tam += 1
            if best.distance < 0.75*second.distance:
                good[i]+=1
        if mejorCoincidencia < good[i] :
            mejorCoincidencia = good[i]
            mejorIndex = i
        print("Coincidencia con el modelo ",i," del ",good[i]/tam*100,"% kp coincidentes: ",good[i]," total: ",tam)
    if (good[mejorIndex]/tam*100 > sensibilidad) :
        print("Mejor modelo: ",mejorIndex)
        cv.imshow("Tomada", frame)
        cv.imshow("Modelo", modelosIM[mejorIndex])
    else:
        print("No hay coincidencia")


sensibilidad = 10
capturado = False                                                   #Determina si se ha guardado o no un primer modelo
modelosKP = []                                                      #Lista de los keypoints de los distintos modelos
modelosIM = []                                                      #Lista de las distintas imágenes que conforman los modelos
bf = cv.BFMatcher()                                             
method = cv.xfeatures2d.SIFT_create()                               #Detector de keypoints elegido
cap = cv.VideoCapture(0)                                            #Capturando la entrada de la cámara
cv.namedWindow("webcam")
while(True):
    k = cv.waitKey(1) & 0xFF                                        #Entrada por teclado
    ret, frame = cap.read()                                         #Lee el frame actual
    if k== 27: break                                                #Si detecta la tecla escape termina la ejecución
    elif k== ord('c') :                                             #Si detecta un espacio captura el frame actual como modelo      
        capturaModelo(frame)
        capturado=True                                              #Capturado está a true pues ya hay almenos un modelo
    elif k== ord(' ') and capturado:
        comparar(frame)           
    else: cv.imshow('webcam',frame)                                 #Muestra el frame actual
cv.destroyAllWindows() 
