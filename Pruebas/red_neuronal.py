import numpy as np
from numpy import asarray
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os
import random


n=4
categorias = []
labels = []
imagenes = []
imagenes1 = []
categorias = os.listdir('Dataset//')

x=1
y=0
print("Leyendo imagenes ...............")
for drectorio in categorias:
    print("Directorio",drectorio)
    for imagen in os.listdir('Dataset//'+drectorio):
        img2 = Image.open('Dataset//' + drectorio + "//" + imagen).resize((28,28))
        print("imagen abierta",img2)
        img2 = np.array(img2)
        print("Imagenes de carpeta ...",img2)

        imagenes.append(img2)
        img3 = Image.open('Dataset//' + drectorio + '//' + imagen).resize((200,200))
        img3 = np.array(img3)
        imagenes1.append(img3)

        labels.append(x)
        if y==599:
            break
        y+=1
    y=0
    if x==n:
        break
    x+=1
labels = np.array(labels)

imagenes = np.array(imagenes)
imagenes1 = np.array(imagenes1)
imagenes = imagenes[:,:,:,0]

large = np.size(labels)
max_images = int(np.size(labels)/n)
print("Numero de categorias ...........",n)
print("Imagenes por categoria .........",str(max_images))
print("Total de imagenes")
print("para el entrenamiento ..........",str(np.size(labels)))

# ...... Escritura y ordenamiento de datos para el entrenamiento .......
archivo = open("Data/frutas.csv","w")
print("")
print("Ordenando imagenes ...............")
for j in range(np.size(imagenes[:,0,0])):
    for k in range(28):
        for l in range(28):
            pixels = imagenes[j,k,l]/255
            archivo.write(str(pixels))

            if k<27 or l<27:
                archivo.write(",")
    archivo.write("\n")
archivo.close()
print("Ordenamiento de los datos realizado")
print("")
# .....................................................................

print("Entrando en la red neuronal ........")
data_path = "Data/frutas.csv"
P = np.loadtxt(data_path,delimiter = ",")
P = np.array(P, dtype='uint8') 

P = P.T

size1 = np.size(P[0,:])
size2 = int(size1/n)

n_muestras = 28*28
Z = np.vstack([P,np.ones((1,size1))])

#Valores esperados
T = np.vstack([np.ones((1,size2)),-np.ones((n-1,size2))])

for i in range(1,n):
    T = np.hstack([T,np.vstack([-np.ones((i,size2)),np.ones((1,size2)),-np.ones((n-1)-i,size2)])])

#Red neuronal Adaline
R = np.dot(Z,Z.T)/size1
H = np.dot(Z,T.T)/size1
X = np.linalg.pinv(R)@H

W = X[0:n_muestras,:].T

b = X[n_muestras,:].reshape(-1,1)
# ...............................................................
index = np.ones((1,size1))
neurona_sal = np.ones((1,size1))

print("Iniciando entrenamiento .............")
for q in range(size1):
    a = np.dot(W,P[:,q]).reshape(-1,1)+b
    neurona_sal[:,q] = p.amax(a)
    posicion = np.where(a == neurona_sal[:,q])
    index[:,q] = posicion[0]

print("Entrenamiiento finalizado ...........")
#Valores reales
y = np.zeros((1,size2))

for j in range(1,n):
    y = np.hstack([y,j*np.ones((1,size2))])

numero_aciertos = np.sum(y==index)
porcentaje_aciertos = (numero_aciertos/size1)*100

print("Total de frutas identificadas ......"+str(numero_aciertos))
print("Porcentaje de acertividad .........."+str(porcentaje_aciertos)+"%")
print("")
# ................................................................
print("Testeo de la red neuronal")
num_row = 4
num_col = 5
fig,axes = plt.subplots(num_row,num_col,figsize=(2*num_col,1.5*num_row))
plt.tight_layout()
n=0
for k in range(20):

    indice = np.round((size1-1)*np.random.rand(1),0)
    numero_reconocido = index[:,int(indice)]

    ax = plt.subplot(num_row,num_col,n+1)
    ax.imshow(imagenes1[int(indice),:,:])

    if numero_reconocido == 0:
        ax.set_title("Fruta no reconocida")
    elif numero_reconocido == 1:
        ax.set_title("Manzana roja")
    elif numero_reconocido == 2:
        ax.set_title("Platano")
    elif numero_reconocido == 3:
        ax.set_title("Mandarina")
    if n==(k/num_row):
        n=0
    n+=1
plt.show()

print("Iniciando video ..............")
captura = cv2.videoCapture(0)
i = 0

if not captura.isOpened():
    print("No se puede abrir la camara")
    exit()

tam_img=28

while(captura.isOpened()):
    #Captura trama a trama
    ret, frame = captura.read()
    if ret == false:
        break
    frame2 = frame

    frame = cv2.resize(frame,(tam_img,tam_img), interpolation = cv2.INTER_LINE)
    frame = frame[:,:,0]
    frame = np.array(frame)
    frame = frame.reshape(-1,1)
    a = np.dot(W,frame).reshape(-1,1)+b #Red neuronal
    
    neurona_sal1 = np.amax(a)

    posicion = np.where(a==neurona_sal1)
    numero_reconocido = int(posicion[0])
    if numero_reconocido == 0:
        fruta = "Fruta no reconocida"
        x=100
    elif numero_reconocido == 1:
        fruta = "Manzana roja"
        x=210
    elif numero_reconocido == 2:
        fruta = "Platano"
        x=250
    elif numero_reconocido == 3:
        fruta = "Mandarina"
        x=290
    #print(numero reconocido)
    escala_grises = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    escala_grises = cv2.putText(frame2,fruta,(x,50),cv2.FONT_HERSHEY_SIMPLEX,)