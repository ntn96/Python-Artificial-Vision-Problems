#!/usr/bin/env python

import cv2 as cv                                                    #Antonio Serrano Miralles
import numpy as np
import skimage.io  as io

def rgb2gray(x):
    return cv.cvtColor(x,cv.COLOR_RGB2GRAY)

# crea un vector (array 1D)
def vec(*argn):
    return np.array(argn)

# juntar columnas
def jc(*args):
    return np.hstack(args)

# aplica una transformación homogénea h a un conjunto
# de puntos ordinarios, almacenados como filas 
def htrans(h,x):
    return inhomog(homog(x) @ h.T)

# convierte un conjunto de puntos ordinarios (almacenados como filas de la matriz de entrada)
# en coordenas homogéneas (añadimos una columna de 1)
def homog(x):
    ax = np.array(x)
    uc = np.ones(ax.shape[:-1]+(1,))
    return np.append(ax,uc,axis=-1)

# convierte en coordenadas tradicionales
def inhomog(x):
    ax = np.array(x)
    return ax[..., :-1] / ax[...,[-1]]

# intenta detectar polígonos de n lados
def polygons(cs,n,prec=2):
    rs = [ redu(c,prec) for c in cs ]               #reduciomos el número de nodos de acuerdo a la precisión introducida
    return [ r for r in rs if r.shape[0] == n ]     #Devolvemos aquellos polígonos que tengan n lados

# detecta siluetas oscuras
def extractContours(g, minarea=10, minredon=25, reduprec=1):
    ret, gt = cv.threshold(g,189,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    
    contours = cv.findContours(gt, cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)[-2]

    h,w = g.shape
    
    tharea = (min(h,w)*minarea/100.)**2 
    
    def good(c):
        oa,r = redondez(c)
        black = oa > 0 # and positive orientation
        return black and abs(oa) >= tharea and r > minredon

    ok = [redu(c.reshape(-1,2),reduprec) for c in contours if good(c)]
    return [ c for c in ok if internal(c,h,w) ]

# area, con signo positivo si el contorno se recorre "counterclockwise"
def orientation(x):
    return cv.contourArea(x.astype(np.float32),oriented=True)

# ratio area/perímetro^2, normalizado para que 100 (el arg es %) = círculo
def redondez(c):
    p = cv.arcLength(c.astype(np.float32),closed=True)
    oa = orientation(c)
    if p>0:
        return oa, 100*4*np.pi*abs(oa)/p**2
    else:
        return 0,0

def kgen(sz,f):     #Devuelve la matriz de calibración
    w,h = sz
    w2 = w / 2
    h2 = h / 2
    return np.array([[f*w2, 0,    w2],
                     [0,    f*w2, h2],
                     [0,    0,    1 ]])

# reducción de nodos
def redu(c,eps=0.5):
    red = cv.approxPolyDP(c,eps,True)
    return red.reshape(-1,2)

def boundingBox(c):
    (x1, y1), (x2, y2) = c.min(0), c.max(0)
    return (x1, y1), (x2, y2)

# comprobar que el contorno no se sale de la imagen
def internal(c,h,w):
    (x1, y1), (x2, y2) = boundingBox(c)
    return x1>1 and x2 < w-2 and y1 > 1 and y2 < h-2

# mide el error de una transformación (p.ej. una cámara)
# rms = root mean squared error
# "reprojection error"
def rmsreproj(view,model,transf):
    err = view - htrans(transf,model)
    return np.sqrt(np.mean(err.flatten()**2))

def pose(K,image,model):
    ok,rvec,tvec = cv.solvePnP(model,image,K,vec(0,0,0,0))      #Obtengo los vectores de rotacion y transltacion gracias a la función de OpenCV
    if not ok:
        return 1e6, None
    R,_ = cv.Rodrigues(rvec)                                    #Convierte el vector rotacion que obtenemos a una matriz rotacion
    M = K @ jc(R,tvec)                                          #Con la matriz de calibración y juntando por columas la de              
    rms = rmsreproj(image,model,M)                              #rotacion y el vector de translacion 
    return rms, M                                               #Obtengo la matriz real

def rots(c):
    return [np.roll(c,k,0) for k in range(len(c))]

# probamos todas las asociaciones de puntos imagen con modelo
# y nos quedamos con la que produzca menos error
def bestPose(K,view,model):
    poses = [ pose(K, v.astype(float), model) for v in rots(view) ]
    return sorted(poses,key=lambda p: p[0])[0]

# muestra un polígono cuyos nodos son las filas de un array 2D
def shcont(frame, c, color=(255,0,0), nodes=True):              #Con modificaciones personales para este ejercicio
    x = c[:,0]
    y = c[:,1]
    x = np.append(x,x[0])
    y = np.append(y,y[0])
    for index, value in enumerate(x) :                          #Recorre la lista de puntos, asumo que el tamaño de la lista x es igual al de y
        cv.circle(frame,(int(value),int(y[index])),5,color,1)   #Dibujo usando cv los vértices de las figuras
        if nodes and index < len(x)-1:                          #Dibujo las lineas que unen los vértices del poligono
            cv.line(frame, (int(value), int(y[index])),(int(x[index+1]),int(y[index+1])),color,1)


cv.namedWindow("resultado")
cap = cv.VideoCapture(1)                #Atención, aquí tengo uno para usar mi webcam externa y no la incorporada en mi portatil
marker = np.array([ [ 0. ,  0. ,  0. ], #Aquí se guarda la estructura de los puntos que conforman la figura del marcador
                    [ 0. ,  1. ,  0. ], #La L achatada a la altura
                    [ 0.5,  1. ,  0. ],
                    [ 0.5,  0.5,  0. ],
                    [ 1. ,  0.5,  0. ],
                    [ 1. ,  0. ,  0. ]])
cube = np.array([                       #El objeto de prueba que utilizaré que es un cubo
    [0,0,0],
    [1,0,0],
    [1,1,0],
    [0,1,0],
    [0,0,0],
    
    [0,0,1],
    [1,0,1],
    [1,1,1],
    [0,1,1],
    [0,0,1],
        
    [1,0,1],
    [1,0,0],
    [1,1,0],
    [1,1,1],
    [0,1,1],
    [0,1,0]
    ])
K = kgen((640,480),1.6)                 #Usaré la matriz de calibración más común con el valor de f de 1.6 y el tamaño de la imagen
while(True):                            #ajustada al tamaño del frame
    k = cv.waitKey(1) & 0xFF        
    if k == 27: break                   #Detecta la tecla Esc para salir de la aplicación

    ret, frame = cap.read()
    grayFrame = rgb2gray(frame)         #Pasamos los frames a escala de grises
    conts = extractContours(grayFrame,reduprec=3)   #Obtengo los contornos
    good = polygons(conts,6)                        #y los filtro para obtener solo los que se correspondan con un poligono de 6 vertices
    if len(good) > 0 :                              #Si hay algún resultado
        err,Me = bestPose(K,good[0],marker)         #Busco la mejor manera de colocar el marcador en los poligonos detectados
        shcont(frame,htrans(Me,marker),color=(0,0,255))             #Dibujamos el marcador de acuerdo a los resultados
        shcont(frame,htrans(Me,cube/2),color=(255,0,0))  #Dibujamos la figura
    cv.imshow('resultado', frame)                   #Muestro el frame resultado

cv.destroyAllWindows()                  #Para el fin de la ejecución liberamos la ventana
