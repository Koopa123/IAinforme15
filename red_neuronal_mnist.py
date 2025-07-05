# red_neuronal_mnist.py
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# Cargar el dataset MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalización
x_train, x_test = x_train / 255.0, x_test / 255.0

# Codificación one-hot de las etiquetas
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Crear el modelo
model = Sequential([
    Flatten(input_shape=(28, 28)),       # Aplanar imagen 28x28 a vector
    Dense(128, activation='relu'),       # Capa oculta con 128 neuronas
    Dense(10, activation='softmax')      # Capa de salida (10 clases)
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

# Evaluar en el conjunto de prueba
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Precisión en test: {test_acc:.4f}")
