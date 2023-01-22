import cv2
import os

input_images_path = "Dataset/Manzana_roja"
files_names = os.listdir(input_images_path)

output_images_path = "Dataset/Manzana"
if not os.path.exists(output_images_path):
    os.makedirs(output_images_path)
    print("Directorio creado: ", output_images_path)

count = 0
for file_name in files_names:
    if file_name.split(".")[-1] not in ["jpeg", "png", "jpg"]:
        continue
    image_path = input_images_path + "/" + file_name
    image = cv2.imread(image_path)
    if image is None:
        continue
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(output_images_path + "/image" + str(count) + ".jpg", image)
    count += 1

print("Tamaño de todas las imágenes cambiado a 224x224 y guardadas en: ", output_images_path)