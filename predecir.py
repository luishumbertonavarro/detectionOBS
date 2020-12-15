import numpy as np
import tensorflow as tf
print(tf.__version__)


from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.models import load_model



longitud, altura = 100, 100
modelo = './modelo/modelo.h5'
pesos_modelo = './modelo/pesos.h5'
cnn = load_model(modelo)
cnn.load_weights(pesos_modelo)


def predict(file):
    x = load_img(file, target_size=(longitud, altura))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    array = cnn.predict(x)
    result = array[0]
    answer = np.argmax(result)
    if answer == 0:
        print('palma')
    elif answer == 1:
        print('paz')
    elif answer == 2:
        print('pulgar')
    return answer


predict('palmanga.jpg')
