import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt

IMAGE_SIZE = 256
BATCH_SIZE = 32

train_path = '/kaggle/input/fruit-recognition/train/train'

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_path,
    seed = 99,
    shuffle = True,
    image_size = (IMAGE_SIZE, IMAGE_SIZE),
    batch_size = BATCH_SIZE
)

class_names = dataset.class_names
class_names

len(dataset)
527 * 32
# Total Image we have is 16854, and we have a set of (batch) 32.
# So if we multiply the length of dataset by batch size will get all images.

for image_batch, labels_batch in dataset.take(1): # take first batch.
    print(image_batch.shape)
    print(labels_batch.numpy())
for image_batch, labels_batch in dataset.take(1): # take first batch.
#     print(image_batch[0])
    print(image_batch[0].numpy())

plt.figure(figsize = (10,10))
for image_batch, labels_batch in dataset.take(1): # take first batch.
    for i in range(12):
        ax = plt.subplot(3,4,i+1)
        plt.imshow(image_batch[i].numpy().astype("uint8"))
        plt.title(class_names[labels_batch[i]])
        plt.axis("off")

def get_dataset(ds, train_split=0.8, val_split = 0.1, test_split = 0.1, shuffle = True, shuffle_size = 1000):
    assert(train_split + test_split + val_split) == 1
    
    ds_size = len(ds)
    if shuffle:
        df = ds.shuffle(shuffle_size, seed = 12)
        
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
#     test_size = int(test_split * ds_size)
    
    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    
    return train_ds, val_ds, test_ds   


train_ds, val_ds, test_ds = get_dataset(dataset)

len(train_ds), len(val_ds), len(test_ds)

resize_rescale = tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    layers.experimental.preprocessing.Rescaling(1./255)
])

data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.2),
])

# apply data augmentation on training data
train_ds = train_ds.map(
    lambda x, y : (data_augmentation(x, training = True), y)
).prefetch(buffer_size = tf.data.AUTOTUNE)

CHANNELS = 3
input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes = len(class_names)
model = models.Sequential([
    resize_rescale,
    layers.Conv2D(32, kernel_size = (3,3), activation = 'relu', input_shape = input_shape),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, kernel_size = (3,3), activation = 'relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, kernel_size = (3,3), activation = 'relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, kernel_size = (3,3), activation = 'relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, kernel_size = (3,3), activation = 'relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, kernel_size = (3,3), activation = 'relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(n_classes, activation='softmax'),
])
model.build(input_shape = input_shape)
model.summary()
model.compile(
    optimizer = 'adam',
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False),
    metrics = ['accuracy']
)
hist = model.fit(
    train_ds,
    batch_size = BATCH_SIZE,
    validation_data = val_ds,
    verbose = 1,
    epochs = 35,
    
)
scores = model.evaluate(test_ds)
scores
hist
hist.params
hist.history.keys()
type(hist.history['loss'])
len(hist.history['loss'])
acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']

loss = hist.history['loss']
val_loss = hist.history['val_loss']
EPOCHS = 35
plt.figure(figsize = (8,8))
plt.subplot(1,2,1)
plt.plot(range(EPOCHS), acc, label = 'Training Accuracy')
plt.plot(range(EPOCHS), val_acc, label = 'Validation Accuracy')
plt.legend(loc = 'lower right')
plt.title("Training and Validation Accuracy")

plt.subplot(1,2,2)
plt.plot(range(EPOCHS), loss, label = 'Training Loss')
plt.plot(range(EPOCHS), val_loss, label = 'Validation Loss')
plt.legend(loc = 'upper right')
plt.title("Training and Validation Loss")
import numpy as np
for images_batch, labels_batch in test_ds.take(1):
    first_image = images_batch[0].numpy().astype('uint8')
    first_label = labels_batch[0].numpy()
    
    print("First Image to Predict")
    plt.imshow(first_image)
    print("Actual Label : ", class_names[first_label])
    
    batch_prediction = model.predict(images_batch)
    print("Predicted Label : ", class_names[np.argmax(batch_prediction[0])])
def funcPredict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(images[i].numpy())
    img_array = tf.expand_dims(img_array, 0)
    
    pred = model.predict(img_array)
    
    pred_class = class_names[np.argmax(pred[0])]
    confidence = round(100 * (np.max(pred[0])), 2)
    
    return pred_class, confidence
plt.figure(figsize = (15, 15))

for images, labels in test_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
                   
        predicted_class, confidence = funcPredict(model, images[i].numpy())
        actual_class = class_names[labels[i]]
                   
        plt.title(f"Actual : {actual_class},\n Predicted : {predicted_class}. \n confidence : {confidence}%")
        plt.axis("off")
# import os
# model_version = max([int(i) for i in os.listdir("models") + [0]])+1
# model.save(f"models/model")


model.save("models/fruits.h5")

 