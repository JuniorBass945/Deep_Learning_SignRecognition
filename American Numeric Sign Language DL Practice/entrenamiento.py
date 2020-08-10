# coding: utf-8
import numpy as np
import os

from keras import layers, models
from keras.utils import to_categorical
from PIL import Image
from sklearn.model_selection import train_test_split

# Se deshabilitan los warnings de Tensorflow.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

x_data, y_data = [], np.array([])
datacount = 0

"""
    Se extraen las fotos para el entrenamiento y almacenan
    en la variable "x_data". Cada foto se comprime a una
    resolución de 100x100 para mantener uniformidad.

    Se crea la variable "y_data" que contendrá los valores
    de las categorías dadas por cada sub-carpeta en el
    directorio principal donde se encuentra la data.
    "y_data" debe tener el mismo tamaño que la cantidad que
    hay de imágenes.

    Origen de la data usada para el entrenamiento:
    https://www.kaggle.com/muhammadkhalid/sign-language-for-numbers
"""

for number in range(0, 10):
    path = f'data/Sign Language for Numbers/{number}'
    dir_files = os.listdir(path)
    dir_files_count = len(dir_files)

    for f in dir_files:
        # Convert ('L') read in and convert to greyscale.
        img = Image.open(path + f'/{f}').resize((100, 100)).convert('L')
        img = np.array(img)

        x_data.append(img)

    y_data = np.append(y_data, np.full((dir_files_count, 1), str(number)))

"""
    A "x_data" se le agrega una quinta dimensión para el
    próximo entrenamiento y divide entre 255 para transformar
    de pixeles (RGB) a una escala del 0 al 1, aún estableciendo
    la intensidad del color.

    A "y_data" se le transforma de un array que contiene la
    categoría correspondiente a la imagen a una matriz
    dispersa describiendo en sus columnas la categoría
    específica, con en sus celdas un 0 si la foto no pertenece
    a esa categoría y un 1 si lo hace.
"""

x_data = np.array(x_data)
x_data = x_data.reshape((-1, 100, 100, 1))
x_data = np.divide(x_data, 255)

y_data = y_data.reshape(-1, 1)
y_data = to_categorical(y_data)

"""
    Se divide la data en tres partes, una para entrenamiento
    ("X_train" y "y_train"); una para validación
    ("X_validate" y "y_validate"); y una para las métricas
    finales.

    Se establece un modelo secuencial de Keras con una
    cantidad arbitraria de capas (layers), se compila,
    entrena y evalúa el modelo.
"""

print("Entrenando....")

(X_train, X_further,
 y_train, y_further) = train_test_split(x_data, y_data, test_size=0.2)
(X_validate, X_test,
 y_validate, y_test) = train_test_split(X_further, y_further, test_size=0.5)


model = models.Sequential()
model.add(layers.Conv2D(32, (5, 5), strides=(2, 2),
                        activation='relu', input_shape=(100, 100, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10,
          batch_size=64, verbose=1,
          validation_data=(X_validate, y_validate))

[loss, acc] = model.evaluate(X_test, y_test, verbose=1)
print(f"Entrenamiento listo. Precisión del modelo: {acc}.")

"""
    Se guarda el modelo y asegura que este se guardó bien.
"""

print("Guardando modelo.")

model.save('number_hand_gesture_model.h5', save_format='h5')
model.save('number_hand_gesture_model')

model_2 = models.load_model('number_hand_gesture_model/')
assert model_2.get_weights()[0].shape[0] > 0
