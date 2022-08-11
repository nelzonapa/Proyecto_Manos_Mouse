from platform import release
from tkinter import Frame
from turtle import heading, shape
import cv2
import mediapipe as mp
import numpy as np
import pyautogui

mp_drawing=mp.solutions.drawing_utils #Para dibujar el resultado de las detecciones(los 21 puntos y sus conecciones)
mp_hands=mp.solutions.hands

#Especial para vídeo, importante, para usar en este caso la WebCam:
cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)

#Definimos el color_mouse_pointer:
color_mouse_pointer=(255,255,0)

"""Puntos de la pantalla"""
SCREEN_X_INI=0
SCREEN_Y_INI=0
#puntosfinales se agrega las medidas puestas en area_screenshot.py
SCREEN_X_FIN=0+1920
SCREEN_Y_FIN=0+1080

"""
Ahora poondremos el (relacion de aspecto)Aspect Ratio=Ancho/Altura
Esto nos servirá más adelante para poder dibujar un pequeño cuadro en el
vídeo streaming y que tenga la misma relación de aspecto de los puntos 
ya definidos anteriormente en la pantalla
"""
aspect_ratio_screen=(SCREEN_X_FIN-SCREEN_X_INI)/(SCREEN_Y_FIN-SCREEN_Y_INI)
print("aspect_ratio_screen: ",aspect_ratio_screen) #imprimimos aspect radio

X_Y_INI=100 #Con esto indicamos los 100 pixeles que se tendrá en el eje Y y X, para el lado izquierdo y derecho de la pantalla

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,# Cambiamos a 1, porque para el mouse solo necesitamos una
    min_detection_confidence=0.5) as hands:#0.5 por defecto

    while True:
        #leemos el vídeo Stream
        ret, frame=cap.read()
        if ret==False:
            break
        height,width,_=frame.shape #obtenemos el alto y ancho
        frame=cv2.flip(frame,1) #Volteamos la imagen de forma horizontal para visualización tipo espejo

        #Construiremos nuestra AREA para el streaming
        """
        Recordemos que para poder tener de manera correcta el area de 
        streaming, debemos de tener en cuenta que la mano no debe de salir
        del alcance de la cámara.
        """
        #Ancho:
        area_width=width-X_Y_INI*2 #*2 es para que el ancho quede centrado 
        #Alto:
        area_height=int(area_width/aspect_ratio_screen)
        #imagen auxiliar
        aux_image=np.zeros(frame.shape,np.uint8)
        #para graficar y ver que funcione
        aux_image=cv2.rectangle(aux_image,(X_Y_INI,X_Y_INI),(X_Y_INI+area_width,X_Y_INI+area_height),(255,0,0),-1)#ponemos color azul al área
        # cv2.imshow("aux_image",aux_image) #Con este comando probamos que este funcionando
        output=cv2.addWeighted(frame,1,aux_image,0.7,0)


        #Pasaremos el vídeo captado de BGR a RGB
        frame_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        results=hands.process(frame_rgb)# almacenamos frame_rgb en hands,process

        """
        Para el caso en que no se encuentre alguna mano
        """
        if results.multi_hand_landmarks is not None:# en el caso de encontrar una mano
            for hand_landmarks in results.multi_hand_landmarks:
                #MediaPipeHands: https://google.github.io/mediapipe/solutions/hands
                """
                No necesitaremos mostrar todos los puntos y líneas, solo accedremos al punto 9
                """
                #coordenadas x y y del punto, accediendo
                x=int(hand_landmarks.landmark[9].x*width)
                y=int(hand_landmarks.landmark[9].y*height)

                """
                En esta parte se esta implementando lo que se necesita para dar movimiento al mouse
                - x,(X_Y_INI,X_Y_INI+area_width) indica en si las coordenadas, la seccion que se dibujo para la pantalla del streaming
                - (SCREEN_X_INI,SCREEN_X_FIN) es la ubicación en sí de la pantalla
                """
                #Usando Interpolación lineal(.interp()):
                #coordenadas x e y
                xm=np.interp(x,(X_Y_INI,X_Y_INI+area_width),(SCREEN_X_INI,SCREEN_X_FIN))
                ym=np.interp(y,(X_Y_INI,X_Y_INI+area_height),(SCREEN_Y_INI,SCREEN_Y_FIN))
                #ya se tiene las coordenadas para poder mover el mouse.
                pyautogui.moveTo(int(xm),int(ym))


                #Introducimos el circulo que dibujaremos
                #Cambiamos de frame a output
                cv2.circle(output,(x,y),10,color_mouse_pointer,3)# el color_mouse_pointer, esta definido en la parte superior
                cv2.circle(output,(x,y),10,color_mouse_pointer,-1)# el color_mouse_pointer, esta definido en la parte superior

        cv2.imshow("Frame",frame)

        #imagen del output
        cv2.imshow("output",output)

        if cv2.waitKey(1) & 0xFF==27:
            break

cap.release()
cv2.destroyAllWindows