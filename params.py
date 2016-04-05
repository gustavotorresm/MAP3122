# # # # # # # # # # # # # # # # # # # #
# Parâmetros para o cálculo da curva  #
# # # # # # # # # # # # # # # # # # # #
import numpy as np

def k():
	return 1

def V(t):
	return np.array((0, t))

def dV(t):
	return np.array((0, 1))