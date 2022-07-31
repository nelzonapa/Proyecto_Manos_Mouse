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
    puesto por x,y,z que contienen decimales.
    """
    print("Hand landmarks: ",results.multi_hand_landmarks)

    #volteamos nuevamente para dejarla con la orientación horiginal
    image=cv2.flip(image,1)
    #Visualizar la imagen
cv2.imshow("Image",image)
cv2.waitKey(0)
cv2.destroyAllWindows()

