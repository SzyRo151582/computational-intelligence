import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras import datasets
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix
from tensorflow.python.keras.callbacks import History

# Load dataset
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Preprocess data
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1)).astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
original_test_labels = np.argmax(test_labels, axis=1)  # Save original labels for confusion matrix

# Define model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = History()
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2, callbacks=[history])

# Evaluate on test set
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc*100:.2f}%")

# Predict on test images
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# Confusion matrix
cm = confusion_matrix(original_test_labels, predicted_labels)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Plotting training and validation accuracy
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True, linestyle='--', color='grey')
plt.legend()

# Plotting training and validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', color='grey')
plt.legend()

plt.tight_layout()
plt.show()

# Display 25 images from the test set with their predicted labels
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i].reshape(28,28), cmap=plt.cm.binary)
    plt.xlabel(predicted_labels[i])
plt.show()


# a) Funkja reshape zmienia wymiary grafiki, to_categorical przekształca wartość na macierz biniarną, a np.argmax
# sumuje etykiety - na przykład cyrf 1 jest tyle, 2 tyle itd.

# b) 1 warstwa otrzymuje grafike na wejściu o rozmiarach 28x28 pikseli, przekształca ją za pomocą jądra o kszatłcie 3x3
# i daje następnej warstwie
# 2 warstwa otrzymuje nowe obrazki, pomniejsza je i przesyła dalej
# 3 warstwa spłaszcza otrzymane grafiki do bardzo małych i przesyła dalej
# 4 warstwa otrzymuje graiki w formie inputu i przedtwarza je przez warstwę neuronów
# 5 i ostatnia warstwa powinna wydac pozytywny werdykt, jaka liczba była na grafice

# c) Często mylił cyfrę 9 z 4, 7 lub 8

# d) W tym przypadku raczej sieć się dobrze nauczyła. Myślę, że gdyby dodać więcej epok sieć mogła by się już przeuczyć.

# e) Stworzyć pętlę iterującą przez liczbę epok, którą ustali użytkownik. Przy każdym wykonaniu uczenia się
# ma przypisac do zmiennych o wyglądzie tablicy wyniki i porównać elementy tablicy. Jeśli wartość uczenia jest
# lepsza niż poprzednia, zapisz model
