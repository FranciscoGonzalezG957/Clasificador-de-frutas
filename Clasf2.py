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

images = []
labels = []
for i, category in enumerate(categories):
    for image_name in os.listdir(path + category):
        image = cv2.imread(path + category + '/' + image_name)
        image = cv2.resize(image, (50,50))
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        images.append(image)
        labels.append(i)
X = np.array(images)
Y = np.array(labels)

#Dividir el nuero de imagenes y etiquetas en un conjunto de entrenemiento y un conjunto de prueba
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

#Convertir las etiquetas a un vector de categorias
y_train = tf.keras.utils.to_categorical(y_train, 4) # Change the number of categories here
y_test = tf.keras.utils.to_categorical(y_test, 4) # Change the number of categories here

model = keras.Sequential()
model.add(layers.Conv2D(32,(3,3),activation = 'relu', input_shape = (50,50,3))) # Change the input shape here
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(4, activation = 'softmax')) # Change the number of categories here

#Compilacion del modelo
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#Entrenamiento del modelo
model.fit(x_train, y_train, batch_size = 32, epochs = 10, validation_data = (x_test, y_test))

#Evaluacion del modelo
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('Test accuracy:', test_acc)

#Capturar imagen en tiempo real
cap = cv2.VideoCapture(0)

redBajo1 = np.array([0, 100, 100], np.uint8)
redAlto1 = np.array([10, 255, 255], np.uint8)
orangeBajo = np.array([10, 250, 100], np.uint8)
orangeAlto = np.array([20, 255, 255], np.uint8)
yellowBajo = np.array([28, 100, 100], np.uint8)
yellowAlto = np.array([32, 255, 255], np.uint8)
greenBajo = np.array([40, 150, 100], np.uint8)
greenAlto = np.array([75, 255, 255], np.uint8)

while True:
    ret, frame = cap.read()
    #Preprocesar imagen
    image = cv2.resize(frame, (50,50))
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

    # Detectar el color de la fruta
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    maskRed = cv2.inRange(hsv, redBajo1, redAlto1)
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
    print("La fruta es: ", class_name, " y su madurez es: ", madurez)
    #Mostrar la imagen en pantalla
    cv2.imshow("Captura de fruta", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#liberar la camara
cap.release() 
