from platform import release
from tkinter import Frame
from turtle import heading
import cv2
import mediapipe as mp

mp_drawing=mp.solutions.drawing_utils #Para dibujar el resultado de las detecciones(los 21 puntos y sus conecciones)
mp_hands=mp.solutions.hands

#Especial para vídeo, importante, para usar en este caso la WebCam:
cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,# Aqui es donde definimos cuántas manos detectar
    min_detection_confidence=0.5) as hands:#0.5 por defecto
    while True:
        #leemos el vídeo Stream
        ret, frame=cap.read()
        if ret==False:
            break
        height,width,_=frame.shape #obtenemos el alto y ancho
        frame=cv2.flip(frame,1) #Volteamos la imagen de forma horizontal para visualización tipo espejo

        cv2.imshow("Frame",frame)
        if cv2.waitKey(1) & 0xFF==27:
            break

cap.release()
cv2.destroyAllWindows