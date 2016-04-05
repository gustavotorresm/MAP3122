import numpy as np
from numpy import linalg as alglin
from scipy.special import lambertw as W
import matplotlib
from params import *
import matplotlib.pyplot as plt

tn = tn1 = Un = h = 0

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

def RungeKutta(U):
	k1 = h*f(tn, U)
	k2 = h*f(tn + h/2, U + k1/2)
	return k2

def Real(t):
	n = np.real(W(np.exp(1 - 4*t/10)))
	x = 10 * np.sqrt(n)
	y = 5 * (n - np.log(n) - 1)/2
	return np.array((x, y))

# for j in range(10):
#	print(j, end='\t')

to = 0.
tf = 40.
n  = int(2. ** 7)
h = dt = (tf - to)/n

t = np.empty((n + 1)) # Tamanho n + 1 para que y(n) = y(tf)
t[0] = to

u = np.empty((2, n + 1))
u[:,0] = (10, 0)

#x = np.empty((n,2))
#x[0] = X(0)

tn = to
Un = u[:, 1] = u[:, 0] + RungeKutta(u[:, 0])

tn = t[1] = tn + h

for i in range(2, n + 1):
	tn1 = t[i] = t[i-1] + h
	Un = u[:,i] = FI(u[:,i-1])
	tn = tn1
	#y[i] = y[i-1] + FI(t[i-1], y[i-1], dt)
	#x[i] = X(t[i])

r = Real(t)

# if (j == 5):
#	plt.plot(u[0], u[1])
#	plt.plot(r[0], r[1])

print(alglin.norm(r[:,n-1] - u[:,n-1]))

v = np.empty((2, n + 1))

for i in range(n + 1):
	v[:,i] = V(t[i])

plt.plot(v[0], v[1])
plt.plot(u[0], u[1])
plt.show()