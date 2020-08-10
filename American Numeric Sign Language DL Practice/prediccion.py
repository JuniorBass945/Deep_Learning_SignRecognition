import numpy as np
import os

from keras import models
from PIL import Image

# Se deshabilitan los warnings de Tensorflow.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Se carga el modelo.
model = models.load_model('number_hand_gesture_model/')

"""
    Espacio para evaluar el modelo con data real, tome una
    foto de su mano (preferiblemente de laderecha debido a
    que la data tiene más gestos de manos derecha que de
    la izquierda) y cárgela de la manera que se muestra a
    continuación para evaluar el número que señaló en
    el lenguaje de señas americano.

    Se rota la imagen a -90 grados debido a que la librería
    de Python PIL no reconoce, asumiendo que tomó la imagen
    recta en su celular o cámara, el ángulo en el que tomó
    la foto.
"""

testing = np.array([])

img = Image.open('test/my_seven_3.jpg').convert('L')
img = img.resize((100, 100)).rotate(-90)
testing = np.append(testing, np.array(img))

x_testing = testing.reshape(1, 100, 100, 1)
x_testing = np.divide(x_testing, 255)

numero_predicho = np.argmax(model.predict(x_testing))
print(f"El número predicho es: {numero_predicho}.")
