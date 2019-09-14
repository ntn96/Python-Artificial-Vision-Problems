#!/usr/bin/env python

import cv2 as cv                                                    #Antonio Serrano Miralles
import numpy as np

global ultimaX                                                      #Guarda donde la coordenada x del último lugar en el que se hizo click
global ultimaY                                                      #Guarda donde la coordenada y del último lugar en el que se hizo click
global frame                                                        #El último frame capturado
global lista                                                        #Lista de imágenes de refencia
global listaHisto                                                   #Lista de histogramas de las imágenes de referencia
global modeReference                                                #Está en true cuando estamos en modo recortar imágenes de referencia

def fun(event, x, y, flags, param):                                 #Función a la que se entra con cada evento de ratón
 global ultimaX
 global ultimaY
 global frame
 global lista
 global modeReference
 if event == cv.EVENT_LBUTTONDOWN:                                  #Si se ha pulsado el botón izquierdo
  print(x,y)                                                        #Imprime en consola donde se ha hecho click
  if ultimaX == -1 and ultimaY == -1:                               #Si es el primero de los dos clicks necesario para delimitar una ROI
   ultimaX = x                                                      #Se actualiza los valores de las ultimas coordenadas x e y
   ultimaY = y
  else:
   reg = frame[ultimaY:y,ultimaX:x]                                 #Guarda el fragmento de la imagen recortado
   if modeReference:                                                #Si nos encontramos en modo de recortar imágenes de referencia
    lista.append(reg)                                               #Guardo la imagen en la lista
    print('número de regerencias guardadas: ',len(lista))           #Imprimo el número de imágenes que tenemos como referencia
    cv.imshow('ultima referencia',cv.resize(reg,(256,256)))         #Muestro las imágenes con un tamaño por defecto de 256*256
    hist = calculateHist(reg)                                       #Calculo el histograma de colores para la imagen de referencia actual
    cv.imshow('histograma referencia',cv.resize(hist,(256,256)))    #Muestro el histograma en 256*256
    listaHisto.append(hist)                                         #Añado el histograma a la lista de histogramas de referencia
    print('número de histogramas guardados: ',len(listaHisto))      #Imprimo el número de histogramas de imagens de referencia 
   else:
    print('modo referencia desactivado')
    histFrame = calculateHist(reg)                                  #Calculo el histograma del frame en cuestión
    cv.imshow('histograma ROI',cv.resize(histFrame,(256,256)))      #Lo muestro en 256*256
    imgParecida = compareHist(histFrame)                            #Llamo a la función de comparación con el histograma actual y me devuelve
    cv.imshow('imagen tomada',cv.resize(reg,(256,256)))             #La imagen de refencia más parecida
    cv.imshow('imagen parecida', cv.resize(imgParecida,(256,256)))
    #comparar reg con lista de ref
   ultimaX = -1                                                     #Reseteamos el valor de las coordenadas para poder coger otro fragmento
   ultimaY = -1

def compareHist(hist):
 global listaHisto
 global lista
 parecido = 10000000000                                             #Variable que almacena el numero de diferencias entre la imagenes
 index = -1                                                         #El índice de la imagen más parecida
 for i in range(0,len(listaHisto)):                                 #Para cada imagen de la lista de referencias
  dif = np.sum(abs(hist-listaHisto[i]),axis=2)                      #Calculamos la diferencia
  suma = np.sum(dif)                                                #Que será un valor entero
  if parecido > suma:                                               #Si la imagen actual es más parecida que la más prometedora 
   parecido = suma                                                  #Actualizamos el valor de parecido y del índice
   index = i
 cv.imshow('histograma referencia',cv.resize(listaHisto[index],(256,256))) #Muestro el histograma de la imagen más parecida
 return lista[index]                                                #Devuelvo la imagen más parecidad de la lista de referencias
  #diferencia de histogramas

def calculateHist(reg):
 global listaHisto                                           #Calculo un histrograma para el canal rojo, otro para el verde y otro para el azul
 h0,b0 = np.histogram(reg[:,:,[0]], bins=256, range=(0,256)) #Red   
 kk0 = np.array([2*b0[1:], 480-h0*(480/10000)]).T.astype(int)
 h1,b1 = np.histogram(reg[:,:,[1]], bins=256, range=(0,256)) #Green
 kk1 = np.array([2*b1[1:], 480-h1*(480/10000)]).T.astype(int)
 h2,b2 = np.histogram(reg[:,:,[2]], bins=256, range=(0,256)) #Blue
 kk2 = np.array([2*b2[1:], 480-h2*(480/10000)]).T.astype(int)
 blank0 = np.zeros((500,520,3), np.uint8)                    #Los paso a imágenes vacias
 blank1 = np.zeros((500,550,3), np.uint8)
 blank2 = np.zeros((500,550,3), np.uint8)
 cv.polylines(blank0, [kk0], isClosed=False, color=(255,0,0), thickness=2)
 cv.polylines(blank1, [kk1], isClosed=False, color=(0,255,0), thickness=2)
 cv.polylines(blank2, [kk2], isClosed=False, color=(0,0,255), thickness=2)
 blank = np.hstack([blank0,blank1,blank2])                   #Concateno los histogramas y lo retorno como una única imagen
 return blank

cap = cv.VideoCapture(0)
cv.namedWindow("webcam")
ultimaX = -1
ultimaY = -1
cv.setMouseCallback("webcam", fun)         #Pongo que los eventos de raton para la ventana webcam sean tratados por la función fun
lista = []                                 #Creo una lista vacía para las imágenes de referencia y sus histogramas
listaHisto = []
modeReference = True                       #Pongo el modo referencia a true porque primero hay que recortar las imágenes de referencia
while(True):
 k = cv.waitKey(1) & 0xFF
 if k== 27: break                          #Cuando pulso escape acaba la ejecución
 elif k== ord(' '): modeReference = False  #Cuando se pulsa espacio se pone el modo comparativo de imágenes
 ret, frame = cap.read()
 cv.imshow('webcam',frame)
cv.destroyAllWindows()
