from os import listdir

from keras.src.utils import load_img, img_to_array
from numpy.ma.core import asarray
# from numpy import savez, load
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.python.keras.callbacks import History

import matplotlib.pyplot as plt

path_file = "dogs-cats-mini\\"
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

# pictures = load("dogs-cats-mini-pictures.npz")
# labels = load("dogs-cats-mini-labels.npz")
print(pictures.shape, labels.shape)

# savez("dogs-cats-mini-pictures", pictures)
# savez("dogs-cats-mini-labels", labels)

pic_train, pic_test, lab_train, lab_test = train_test_split(pictures, labels, test_size=0.3, random_state=1)

model = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=(40, 40, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(filters=128, kernel_size=(3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(units=128, activation="relu"),
    Dense(units=1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
history = History()
model.fit(pic_train, lab_train, batch_size=64, epochs=15, callbacks=[history], validation_split=0.2)

test_accuracy, test_loss = model.evaluate(pic_test, lab_test)
print(f"Test accuracy - {round(test_accuracy * 100, 2)} %.")

predictions = model.predict(pic_test)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True, linestyle='--', color='grey')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', color='grey')
plt.legend()

plt.show()
