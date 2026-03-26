from os import listdir

from numpy.ma.core import asarray

from keras.src.utils import load_img, img_to_array
from keras.src.applications.efficientnet import EfficientNetB1
from sklearn.model_selection import train_test_split

path_file = "C:\\Users\\Stjudent\\Desktop\\UG Informatyka\\1rok\\Inteligencja obliczeniowa\\dogs-cats-mini\\"
pictures, labels = list(), list()

for file in listdir(path_file):
    output = 0.0
    if file.startswith("dog"):
        output = 1.0
    picture = load_img(path_file + file, target_size=(40, 40))
    picture = img_to_array(picture)
    pictures.append(picture)
    labels.append(output)

pictures = asarray(pictures)
labels = asarray(labels)

pic_train, pic_test, lab_train, lab_test = train_test_split(pictures, labels, test_size=0.3, random_state=1)

model = EfficientNetB1(weights='imagenet')
