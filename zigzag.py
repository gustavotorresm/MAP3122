# # # # # # # # # # # # # # # # # # # #
# Parâmetros para o cálculo da curva  #
# # # # # # # # # # # # # # # # # # # #
import numpy as np

def k():
	return 1

def V(t):
	return np.array((t, 2/np.pi * np.arcsin(np.sin(np.pi * t))))

def dV(t):
	#return np.array((1, 2 * np.cos(np.pi * t)/np.sqrt(1 - np.sin(np.pi * t) ** 2)))
	return np.array((1, 1))
