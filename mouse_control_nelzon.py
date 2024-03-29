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

#Función nueva para calcular distancia entre puntos:
def calcular_distancia(x1,y1,x2,y2):
    punto1=np.array([x1,y1])
    punto2=np.array([x2,y2])

    return np.linalg.norm(punto1-punto2) #operación


#Creamos la función que usaremos para la detección del dedo que hará click
def detect_finger_down(hand_landmarks):
    finger_down=False #En un principio será false
    #Ahora definiremos los colores para la distancia base(palma) y distancia thumb(dedo pulgar)
    color_base=(255,255,0)
    color_thumb=(255,0,0)
    #obtenemos los puntos claves:
    #- para la palma:
    x_base_1=int(hand_landmarks.landmark[0].x*width)#landmark[i] depende de los puntos de la muñeca que define mediapipeHands
    y_base_1=int(hand_landmarks.landmark[0].y*height)

    #- para la el otro punto de la palma:
    x_base_2=int(hand_landmarks.landmark[9].x*width)
    y_base_2=int(hand_landmarks.landmark[9].y*height)

    #Para el dedo indice:
    x_thumb=int(hand_landmarks.landmark[8].x*width)#landmark[i] depende de los puntos de la muñeca que define mediapipeHands
    y_thumb=int(hand_landmarks.landmark[8].y*height)

    """Se debe de hallar la distancia entre los puntos anteriores (base_1 con base_2) y (base_1 e thumb)
    Para ello, vamos a crear una función aparte llamada (calcular_distancia)"""
    distancia_base=calcular_distancia(x_base_1,y_base_1,x_base_2,y_base_2)
    distancia_indice=calcular_distancia(x_base_1,y_base_1,x_thumb,y_thumb)
    
    #condicion
    if distancia_indice<distancia_base:
        finger_down=True
        #cambiamos de color a las lineas
        color_base=(0,255,255)
        color_thumb=(0,255,255)

    #Ahora tendremos que visualizar los cuirculos y las respectivas líneas
    cv2.circle(output,(x_base_1,y_base_1),5,color_base,2)
    cv2.circle(output,(x_thumb,y_thumb),5,color_thumb,2)
    cv2.line(output,(x_base_1,y_base_1),(x_base_2,y_base_2),color_base,3)
    cv2.line(output,(x_base_1,y_base_1),(x_thumb,y_thumb),color_thumb,3)

    #finalmente:
    return finger_down


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

                pyautogui.moveTo(int(xm),int(ym))#para movimiento
                
                """Pondremos la acción de click del mouse"""
                #Función que usaremos para la detección
                if detect_finger_down(hand_landmarks):
                    pyautogui.click()


                #Introducimos el circulo que dibujaremos
                #Cambiamos de frame a output
                cv2.circle(output,(x,y),10,color_mouse_pointer,3)# el color_mouse_pointer, esta definido en la parte superior
                cv2.circle(output,(x,y),10,color_mouse_pointer,-1)# el color_mouse_pointer, esta definido en la parte superior

        # cv2.imshow("Frame",frame) #podemos desactivar el frame

        #imagen del output
        cv2.imshow("output",output)

        if cv2.waitKey(1) & 0xFF==27:
            break

cap.release()
cv2.destroyAllWindows