#bibliotecas necesarias
import cv2
import numpy as np
import pyautogui

while True:
    screenshot=pyautogui.screenshot(region=(0,0,1920,1080))
    """
    En region se dan los puntos del área a marcar (x,y)
    -> X es el punto de inicio superior izquierdo a la derecha
    -> Y es el punto de inicio superior izquierdo hacia abajo
    """

    #Necesitamos convertir el "screenshot" aun array de NUMPY
    #para así poder mostrar el "screenshot"
    screenshot=np.array(screenshot)

    """Normalmente el screenshot se hace de otro color, y aplicamos lo siguiente
    para poder obtener el color original"""
    screenshot=cv2.cvtColor(screenshot,cv2.COLOR_RGB2BGR)

    #Visualizamos el "screenshot"
    cv2.imshow("screenshot",screenshot)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cv2.destroyAllWindows()