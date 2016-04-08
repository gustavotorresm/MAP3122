# # # # # # # # # # # # # # # # # # # #
# Parâmetros para o cálculo da curva  #
# # # # # # # # # # # # # # # # # # # #
import numpy as np
from scipy.special import lambertw as W

def k():
	return 1

def V(t):
	return np.array((0, t))

def dV(t):
	return np.array((0, 1))

def Real(t):
	n = np.real(W(np.exp(1 - 4*t/10)))
	x = 10 * np.sqrt(n)
	y = 5 * (n - np.log(n) - 1)/2
	return np.array((x, y))
