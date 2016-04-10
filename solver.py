import numpy as np
from numpy import linalg as alglin
import matplotlib
from rosacea import *
import matplotlib.pyplot as plt

def f(t, U):
	diff = V(t) - U
	return k() * alglin.norm(dV(t)) * diff/alglin.norm(diff)

def g(U):
	return Un - U + (h/2) * (f(tn1, U) + f(tn, Un))

def dg(U):
	vel = alglin.norm(dV(tn1))
	diff = V(tn1) - U
	diff_norm = alglin.norm(diff)
	Jac = np.empty((2,2))
	Jac[0][0] = -1 + h * k() * (vel/2) * ((diff[0] ** 2 - diff_norm ** 2)/(diff_norm ** 3))
	Jac[1][0] = Jac[0][1] = h * k() * (vel/2) * (diff[0] * diff[1]) / (diff_norm ** 3)
	Jac[1][1] = -1 + h * k() * (vel/2) * ((diff[1] ** 2 - diff_norm ** 2)/(diff_norm ** 3))
	return Jac

def FI(U):
	X = U.copy()
	previous = X.copy()
	for i in range(100):
		X += -(alglin.inv(dg(X))).dot(g(X))
		if ((X - previous)/X).max() < 1e-3:
			break
		previous = X
	return X

# Runge Kutta para o cálculo no instante em t1
def RungeKutta(U):
	k1 = h*f(tn, U)
	k2 = h*f(tn + h/2, U + k1/2)
	return k2

def init(k = 10):
	global to
	global tf
	global n
	global h
	global Uo

	to = 0.
	tf = 10.
	n  = int(2. ** k) 
	h  = dt = (tf - to)/n
	Uo = np.array((0,0))

def calcula(manufaturado = False):
	global tn
	global tn1
	global Un

	t = np.empty((n + 1)) # Tamanho n + 1 para que y(n) = y(tf)
	t[0] = to

	u = np.empty((2, n + 1))
	Un = u[:,0] = Real(to) if manufaturado else Uo

	tn = to

	for i in range(1, n + 1):
		tn1 = t[i] = t[i-1] + h
		Un = u[:,i] = FI(u[:,i-1])
		tn = tn1

	if manufaturado:
		r = Real(t)
		erro = alglin.norm(r[:,n-1] - u[:,n-1])
	else:
		erro = None

	return {'U': u, 'T' : t, 'erro': erro}


init(15)
X = calcula()

np.save("ro", X['U'])

t = X['T']
np.save("ro-time", t)

v = np.empty((2, n + 1))

for i in range(len(t)):
	v[:,i] = V(t[i])
np.save('ro-alvo', v)
