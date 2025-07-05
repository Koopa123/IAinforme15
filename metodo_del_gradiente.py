# metodo_del_gradiente.py
import numpy as np

# Definir función de error (parabólica): E(w) = w^2
def error(w):
    return w**2

# Derivada de la función: dE/dw = 2w
def derivada(w):
    return 2 * w

# Inicializar peso
w = 5.0  # empieza lejos del mínimo
tasa_aprendizaje = 0.1
epocas = 20

print("Iteración\tw\t\tError")
for i in range(epocas):
    gradiente = derivada(w)
    w = w - tasa_aprendizaje * gradiente
    print(f"{i+1}\t\t{w:.4f}\t\t{error(w):.4f}")
