# # # # # # # # # # # # # # # # # # # #
# Parâmetros para o cálculo da curva  #
# # # # # # # # # # # # # # # # # # # #
import numpy as np

def k():
	return .85

def V(t):
	x = np.cos(9 * t) * np.cos(t)
	y = np.cos(9 * t) * np.sin(t)
	return np.array((x,y))

def dV(t):
	x = - np.sin(t)*(np.cos(9*t)) - 9 * np.sin(9*t)*np.cos(t)
	y =  np.cos(t)*np.cos(9*t) - 9 * np.sin(t)*np.sin(9*t)
	return np.array((x,y))
