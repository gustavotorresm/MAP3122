# # # # # # # # # # # # # # # # # # # #
# Parâmetros para o cálculo da curva  #
# # # # # # # # # # # # # # # # # # # #
import numpy as np

def k():
	return 1

def V(t):
	return np.array([2 * np.cos(t), np.sin(t)])

def dV(t):
	return np.array((- 2 * np.sin(t), np.cos(t)))
