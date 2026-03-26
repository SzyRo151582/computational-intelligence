from keras import Sequential, layers, utils
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Preprocess the data
# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encode the labels
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y.reshape(-1, 1))

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)

# Define the model
model = Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(y_encoded.shape[1], activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")

# Plot the learning curve
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.grid(True, linestyle='--', color='grey')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.grid(True, linestyle='--', color='grey')
plt.legend()

plt.tight_layout()
plt.show()

# Save the model
model.save('iris_model.h5')

# Plot and save the model architecture
utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# a) Standard scaler przekształca dane, które użytkownik może okreslić
# Według dokumentacji sklearn -
# Standaryzuje cechy poprzez usunięcie średniej i skalowanie do wariancji jednostkowej.
# Wynik standardowy próbki x jest obliczany jako:
# z = (x - u) / s

# b) OneHotEncoder zmienia wprowadzone dane na format binarny (0, 1)
# one-hot to kolumna binarna
# etykiety klas są transformowane względem nazw irysów, jeśli jest to ten konkretny gatunek - zwróć 1

# c) Warstwa wejściowa ma 4 warstwy - pokazuje to model.inputs
# X_train.shape[1] to ilość kolumn w tablicy X, [0] to byłaby ilość wierszy
# Warstwa wyjściowa ma 3 warstwy - model.outputs
# y_encoded.shape[1] to ilość kolumn w tablicy y, związanymi z etykietami

# d) Aktywacja liniowa działa tak samo dobrze, softmax sobie już tak dobrze nie radzi (wyniki między 60-70%)

# e) op: sgd, loss: mse - wyniki gorsze o około 15%
# Zmiana funkcji straty na binary_crossentropy nie zmieniła wyników w widoczny sposób
# op: lion, loss: dice - wyniki też praktycznie z niezauważalnymi różnicami

# f) Metoda fit posiada parametr batch_size, gdzie można dostosować rozmiar partii.
# rozmiar 4 i 8 - validation model loss mocno rośnie z powrotem do wysokich wartości, validation model acuraccy
# można powiedzieć, że "skacze"
# rozmiar 16 - tutaj wyniki są już lepsze - validation model loss utrzymuje się w miare niskim poziomie

# g) (Wersja pierwsza) Wydajność uczenia się na początku mocno rośnie, potem waha się w okolicach 95-100%.
# Model jest w miarę dobrze dopasowany (różnice między wykresami nie są znaczące)

# h) Kod w pliku wczytuje model uczenia się, który był używany wcześniej i uzywa go do dalszego testowania.
# Później aktualizuje wyniki modelu.
