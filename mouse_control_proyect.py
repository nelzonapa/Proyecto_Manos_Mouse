from email.mime import image
from turtle import shape
from unittest import result
import cv2
import mediapipe as mp

mp_drawing=mp.solutions.drawing_utils #Para dibujar el resultado de las detecciones(los 21 puntos y sus conecciones)
mp_hands=mp.solutions.hands

"""
Tenemos 'STATIC_IMAGE_MODE'--> TRUE(Este es para imágenes) or FALSE(para vídeo stream, aplica la detección de la palma)

MAX_NUM_HANDS: Este define cuántas manos se detectará (2 por defecto)

MIN_DETECTION_CONFIDENCE: Valor mínimo de confianza del modelo de detección de manos, para dar a conocer si la detección fue exitosa.(0.5 por defecto)

MIN_TRACKING_CONFIDENCE: Valor mínimo de confianza del modelo de rastreo de los land marks, para que el rastreo de los 21 puntos sea considerado como un éxito(0.5 por defecto).
--> En caso de no serlo, se invocará de nuevo al detector de manos
"""

#explorar las opciones de ejecución
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,# Aqui es donde definimos cuántas manos detectar
    min_detection_confidence=0.5) as hands:
    #Leeremos la imagen con opencv
    image=cv2.imread("mostrando_manos.jpg")
    height, width,_ = image.shape #Obtenemos el alto y el ancho de la imagen
    #volteamos horizontalmente para que se pueda identificar la mano izquierda y la derecha(como un reflejo)
    image=cv2.flip(image,1)

    #Pasaremos la imagen de entrada de brg a rgb para la detección de imágenes
    image_rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    results=hands.process(image_rgb)# Obtenemos 2 salidas multi HANDEDNESS y multi HAND LANDMARKS

    #HANDEDNESS
    # print("Handedness: ",results.multi_handedness)#imprimimos lo que se tiene en multi_handedness
    """
    index:
    score: Que tan bien esta identificada la mano
    Label: Etiqueta que nos dirá si se tiene una Mano izquierda o derecha
    """

    #HAND LANDMARKS
    """
    Coleccion de manos rastreadas y/o detectadas, donde cada mano esta 
    representada por una lista de 21 puntos. Donde cada punto esta com-
    puesto por x,y,z que contienen decimales (coordenadas de la imagen).
    """
    #print("Hand landmarks: ",results.multi_hand_landmarks)

    """
    Para el caso en que no se encuentre alguna mano
    """
    if results.multi_hand_landmarks is not None:# en el caso de encontrar una mano
        #------------------------------------------------------------------
        #Dibujando los puntos y sus conexiones con mediapipe (leida de todos los puntos)
        """
        Recorreremos los 21 puntos por cada mano detectada
        """
        #for hand_landmarks in results.multi_hand_landmarks:
            #print(hand_landmarks)# imprime 21 puntos por cada mano
        """
            Dibujando los 21 puntos de los hand_landmarks con ayuda de mediapipe
            """
            #mp_drawing.draw_landmarks(image,hand_landmarks,mp_hands.HAND_CONNECTIONS)
        """
            Ya se específicó la imagen que se tiene en la variable "imagen" donde se 
            dibujará los 21 puntos de cada mano. Incluso uno puede usar :
            multi HANDEDNESS y multi HAND LANDMARKS para adquirir información de puntos o 
            qué manos se estan detectando en la imagen.
            OJO los colores fueron establecidos automáticamente
            """
        """
            EN EL CASO DE QUERER CAMBIAR EL COLOR:
            mp_drawing.draw_landmarks(
                image,hand_landmarks,mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255,255,0),thickness=4,circle_radius=5).
                mp_drawing.DrawingSpec(color=(255,0,255),thickness=4))
            """
        #---------------------------------------------------------------
        #---------------------------------------------------------------
        """
        Accediendo a los puntos clave(hand landmarks) de acuerdo a su nombre, el cual esta establecido
        por mediapipe: 
        Pulgar-> THUMB_TIP 
        Índice-> INDEX_FINGER_TIP
        Medio-> MIDDLE_FINGER_TIP
        Anular-> RING_FINGER_TIP
        Meñique-> PINKY_TIP
        """
        for hand_landmarks in results.multi_hand_landmarks:
            x1=int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x*width)
            """
            Se Accedió a la coordenada "x" del o de los pulgares
            Se debe multiplicar el valor de la coordenada x del o los pulgares por el ancho de la imagen
            para obtener un numero más grande. Finalmente obtenemos su valor sin decimales con int()

            """
            #Para la coordenada y, lo mismo, solo que multiplicado por el alto de la imagen
            y1=int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y*height)
            #print(x1,y1) #mostramos coordenadas del punto 1 en la terminal

            #PROBAMOS las coordenadas obtenidas, dibujando un circulo en las coordenadas
            cv2.circle(image,(x1,y1),3,(255,0,0),3)

            """
            Ubicando los demás puntos:
            """
            x2=int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x*width)
            y2=int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y*height)

            x3=int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x*width)
            y3=int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y*height)

            x4=int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x*width)
            y4=int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y*height)
            
            x5=int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x*width)
            y5=int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y*height)

            cv2.circle(image,(x2,y2),3,(255,0,0),3)
            cv2.circle(image,(x3,y3),3,(255,0,0),3)
            cv2.circle(image,(x4,y4),3,(255,0,0),3)
            cv2.circle(image,(x5,y5),3,(255,0,0),3)
        #---------------------------------------------------------------
    
    #volteamos nuevamente para dejarla con la orientación horiginal
    image=cv2.flip(image,1)
    #Visualizar la imagen
cv2.imshow("Image",image)
cv2.waitKey(0)
cv2.destroyAllWindows()

