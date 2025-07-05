# backpropagation_xor.py
import numpy as np

# Funciones de activación y su derivada
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1 - x)

# Datos de entrada (XOR)
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

# Salida esperada
y = np.array([[0],
              [1],
              [1],
              [0]])

# Inicialización de pesos
np.random.seed(1)
W1 = np.random.rand(2, 4)  # Pesos capa entrada → oculta (2 entradas, 4 neuronas)
W2 = np.random.rand(4, 1)  # Pesos capa oculta → salida (4 neuronas, 1 salida)

# Entrenamiento
for epoch in range(10000):
    # FORWARD
    l1 = sigmoid(np.dot(X, W1))  # Capa oculta
    l2 = sigmoid(np.dot(l1, W2)) # Capa salida

    # BACKPROPAGATION
    error = y - l2
    d_l2 = error * sigmoid_deriv(l2)
    d_l1 = d_l2.dot(W2.T) * sigmoid_deriv(l1)

    # Actualizar pesos
    W2 += l1.T.dot(d_l2)
    W1 += X.T.dot(d_l1)

# Resultado
print("Predicciones finales:")
print(np.round(l2, 3))
