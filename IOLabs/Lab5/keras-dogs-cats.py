from os import listdir

from keras import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.src.callbacks import History
from keras.src.saving import load_model
from keras.src.utils import load_img, img_to_array

from numpy.ma.core import asarray
from pandas.io.common import file_exists
from sklearn.model_selection import train_test_split
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

pic_train, pic_test, lab_train, lab_test = train_test_split(pictures, labels, test_size=0.3, random_state=1)

if file_exists("cat_dog_model.keras"):
    model = load_model("cat_dog_model.keras", compile=False)
else:
# Wersja z internetu
    model = Sequential([
        Conv2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=(40, 40, 3)),
        Conv2D(filters=32, kernel_size=(3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
        Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(filters=128, kernel_size=(3, 3), activation="relu"),
        Conv2D(filters=128, kernel_size=(3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(units=128, activation="relu"),
        Dropout(rate=0.2),
        Dense(units=64, activation="relu"),
        Dropout(rate=0.2),
        Dense(units=1, activation="sigmoid"),
    ])

# Zmiana aktywacji na sigmoid i softmax - sieć popełnia coraz więcej blędów, im więcej jest epok
# model = Sequential([
#     Conv2D(filters=32, kernel_size=(3, 3), activation="sigmoid", input_shape=(40, 40, 3)),
#     MaxPooling2D(pool_size=(2, 2)),
#     Conv2D(filters=64, kernel_size=(3, 3), activation="sigmoid"),
#     MaxPooling2D(pool_size=(2, 2)),
#     Conv2D(filters=128, kernel_size=(3, 3), activation="sigmoid"),
#     MaxPooling2D(pool_size=(2, 2)),
#     Flatten(),
#     Dense(units=128, activation="sigmoid"),
#     Dense(units=1, activation="softmax")
# ])

# Zmiana struktury - dodanie warstwy dense, zmiana filtrów i wielkości jądra w conv
# model = Sequential([
#     Conv2D(filters=16, kernel_size=(4, 4), activation="relu", input_shape=(40, 40, 3)),
#     MaxPooling2D(pool_size=(2, 2)),
#     Conv2D(filters=32, kernel_size=(4, 4), activation="relu"),
#     MaxPooling2D(pool_size=(2, 2)),
#     Conv2D(filters=64, kernel_size=(4, 4), activation="relu"),
#     MaxPooling2D(pool_size=(2, 2)),
#     Flatten(),
#     Dense(units=64, activation="relu"),
#     Dense(units=32, activation="relu"),
#     Dense(units=1, activation="sigmoid")
# ])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
# Zmiana optymalizatora na lion
# model.compile(optimizer="lion", loss="binary_crossentropy", metrics=['accuracy'])
# Trochę lepsze wyniki
history = History()
model.fit(pic_train, lab_train, batch_size=64, epochs=4, callbacks=[history], validation_split=0.01)

test_accuracy, test_loss = model.evaluate(pic_test, lab_test)
print(f"Test accuracy - {round(test_accuracy * 100, 2)} %.")

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

if file_exists("Accuracy and Loss Training.png"):
    plt.savefig("Accuracy and Loss Re-training.png")
else:
    plt.savefig("Accuracy and Loss Training.png")
plt.show()

model.save('cat_dog_model.keras')

# Według strony, występują zdjęcia, na których może być zwierze w ruchu, więc może być rozmyte
# Może też być zwierzę częściowo przysłonięte przez coś
# Są też grafiki, gdzie jest więcej niż jedno zwierze (na przykład dwa koty, kot z psem)
# To wszystko prawdopodobnie może utrudnic uczenie się sieci
