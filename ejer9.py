#!/usr/bin/env python

import dlib                                         #Antonio Serrano Miralles
predictor_path = "./shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()		    #Coge el detector de caras
predictor = dlib.shape_predictor(predictor_path)	#y el predictor de forma

import cv2          as cv
import numpy        as np
from umucv.stream import autoStream

Puntos = []     #Determina los puntos claves detectados en la captura
comparando = False  #Determina si se está comparando los puntos de la captura con los del stream
for key, frame in autoStream():
    img = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    height = np.size(img, 0)    #Saco el tamaño de la imagen para crear una imagen en negro del mismo tamaño
    width = np.size(img, 1)     #Sobre la que poder dibujar la expresión
    expresion = np.zeros((height,width,3), np.uint8)    #Creo la imagen en negro
    dets = detector(img, 0)
    for k, d in enumerate(dets):
        cv.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (255,0,0) )   #rectangulo que limita el rostro
        shape = predictor(img, d)
        L = []                  #Array que guarda las coordenadas de los punto detectados por el predictor
        for p in range(68):
            x = shape.part(p).x
            y = shape.part(p).y
            L.append(np.array([x,y]))
            cv.circle(frame, (x,y), 2,(255,0,0), -1)   #Dibuja cada punto en la imagen que vamos a mostrar 
            if p != 0 and p!=17 and p!=22 and p!=27 and p!=31 and p!= 36 and p!= 42 and p!=48 :  #Para dibujar el contorno del rostro
                cv.line(frame,(x,y),(L[p-1][0],L[p-1][1]),(0,0,255))
                cv.line(expresion,(x,y),(L[p-1][0],L[p-1][1]),(0,0,255))            #Dibuja también la expresión en otra ventana
            if p == 41 or p== 47:       
                cv.line(frame,(x,y),(L[p-5][0],L[p-5][1]),(0,0,255))                #Cierra el contorno de los ojos
                cv.line(expresion,(x,y),(L[p-5][0],L[p-5][1]),(0,0,255))
            if comparando :                             #Si está comparando la captura con el streaming dibuja una línea que une cada punto
                cv.circle(frame, (x,y), 2,(0,0,255), -1)#Actual con su homólogo de la captura
                cv.line(frame,(x,y),(Puntos[p][0],Puntos[p][1]),(0,255,0)) #Con ello se puede deducir como se ha movido el rostro
        L = np.array(L)
    if key == 99 : #Si se pulsa la tecla C se toma una captura con la que se comparará el streaming
        cv.imshow("captura",frame)  #Se muestra la captura en otra ventana
        print("C PULSADA : CAPTURA REALIZADA")
        Puntos = L; #Se guarda una copia de los puntos claves del rostro de la captura
        comparando = True   #Se pasa al estado de comparación
    cv.imshow("faceInvention",frame)    #Se imprime en las respectivas ventanas
    cv.imshow("expresion", expresion)
