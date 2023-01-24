import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import os
from sklearn.model_selection import train_test_split
import numpy as np

#cargar y preprocesar las imagenes
path = 'Dataset/'
categories = os.listdir(path)

# Cargar el modelo entrenado
model = tf.keras.models.load_model('fruit_classifier.h5')

# Capturar imagen en tiempo real
cap = cv2.VideoCapture(0)

redBajo = np.array([0, 100, 100], np.uint8)
redAlto = np.array([10, 255, 255], np.uint8)
orangeBajo = np.array([10, 250, 100], np.uint8)
orangeAlto = np.array([20, 255, 255], np.uint8)
yellowBajo = np.array([28, 100, 100], np.uint8)
yellowAlto = np.array([32, 255, 255], np.uint8)
greenBajo = np.array([40, 150, 100], np.uint8)
greenAlto = np.array([75, 255, 255], np.uint8)

while True:
    ret, frame = cap.read()

    # Preprocesar imagen
    image = cv2.resize(frame, (50,50))
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

    # Detectar el color de la fruta
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    maskRed = cv2.inRange(hsv, redBajo, redAlto)
    maskOrange = cv2.inRange(hsv, orangeBajo, orangeAlto)
    maskYellow = cv2.inRange(hsv, yellowBajo, yellowAlto)
    maskGreen = cv2.inRange(hsv, greenBajo, greenAlto)

    # Evaluar la madurez de la fruta según el color detectado
    if (np.count_nonzero(maskRed) > 1000): # si hay suficiente color rojo en la imagen
        madurez = "madura"
    elif(np.count_nonzero(maskOrange) > 1000): # si hay suficiente color naranja en la imagen
        madurez = "madura"
    elif(np.count_nonzero(maskYellow) > 1000): # si hay suficiente color amarillo en la imagen
        madurez = "madura"
    elif(np.count_nonzero(maskGreen) > 1000): # si hay suficiente color verde en la imagen
        madurez = "inmadura"
    else:
        madurez = ""

    # Hacer prediccion
    pred = model.predict(np.expand_dims(image, axis=0))
    #Clasificar la imagen
    class_idx = np.argmax(pred, axis=1)[0]
    class_name = categories[class_idx]

    #Mostrar el resultado de la clasificacion y la madurez en pantalla
    cv2.putText(frame, "Fruta: " + class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, "Madurez: " + madurez, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow("Captura de fruta", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la camara
cap.release()
cv2.destroyAllWindows()