# # # # # # # # # # # # # # # # # # # #
# Parâmetros para o cálculo da curva  #
# # # # # # # # # # # # # # # # # # # #
import numpy as np

def k():
	return 1

def V(t):
	x = np.cos(2 * t) * np.cos(t)
	y = np.cos(2 * t) * np.sin(t)
	return np.array((x,y))

def dV(t):
	x = -0.5 * (np.sin(t) + 3 * np.sin(3*t))
	y =  0.5 * (3 * np.cos(3*t) - np.cos(t))
	return np.array((x,y))
