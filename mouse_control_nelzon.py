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

        #Pasaremos el vídeo captado de BGR a RGB
        frame_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        results=hands.process(frame_rgb)# almacenamos frame_rgb en hands,process

        """
        Para el caso en que no se encuentre alguna mano
        """
        if results.multi_hand_landmarks is not None:# en el caso de encontrar una mano
            for hand_landmarks in results.multi_hand_landmarks:#Para obtener los 21 puntos por mano detectada
                mp_drawing.draw_landmarks(frame,hand_landmarks,mp_hands.HAND_CONNECTIONS)
                """
                (draw_landmarks) de mediapipe, dibuja los puntos y sus conexiones
                """
                """
                Después de lo anterior ya puedes experimentar con lo aprendido en la leída de manos mediante imagen
                """
        cv2.imshow("Frame",frame)
        if cv2.waitKey(1) & 0xFF==27:
            break

cap.release()
cv2.destroyAllWindows