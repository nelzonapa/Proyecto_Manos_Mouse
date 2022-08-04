from platform import release
from tkinter import Frame
from turtle import heading
import cv2
import mediapipe as mp

mp_drawing=mp.solutions.drawing_utils #Para dibujar el resultado de las detecciones(los 21 puntos y sus conecciones)
mp_hands=mp.solutions.hands

#Especial para vídeo, importante, para usar en este caso la WebCam:
cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)

#Definimos el color_mouse_pointer:
color_mouse_pointer=(255,255,0)

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

                #Introducimos el circulo que dibujaremos
                cv2.circle(frame,(x,y),10,color_mouse_pointer,3)# el color_mouse_pointer, esta definido en la parte superior
                cv2.circle(frame,(x,y),10,color_mouse_pointer,-1)# el color_mouse_pointer, esta definido en la parte superior

        cv2.imshow("Frame",frame)
        if cv2.waitKey(1) & 0xFF==27:
            break

cap.release()
cv2.destroyAllWindows