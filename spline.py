import numpy as np
import matplotlib.pyplot as plt

def spline(X, A, fpo, fpn):
	n = len(X)-1
	H = np.empty((n + 1))
	Alfa = np.empty((n + 1))
	L = np.empty((n + 1))
	mi = np.empty((n + 1))
	z = np.empty((n + 1))
	C = np.empty((n + 1))
	B = np.empty((n + 1))
	D = np.empty((n + 1))

	for i in range(n):
		H[i] = X[i+1] - X[i]
	Alfa[0] = 3 * (A[1] - A[0])/H[0] - 3 * fpo
	Alfa[n] = 3 * fpn - 3 * (A[n] - A[n-1])/H[n-1]

	for j in range(n - 1):
		i = j + 1
		Alfa[i] = (3/H[i]) * (A[i+1] - A[i]) - (3/H[i-1]) * (A[i] - A[i-1])
	L[0] = 2 * H[0]
	mi[0] = 0.5
	z[0] = Alfa[0] / L[0]

	for j in range(n - 1):
		i = j + 1
	L[i] = 2 * (X[i+1] - X[i-1]) - H[i-1]*mi[i-1]
	mi[i] = H[i]/L[i]
	z[i] = (Alfa[i] - H[i-1] * z[i-1]) / L[i]

	L[n] = H[n-1] * (2 - mi[n-1])
	z[n] = (Alfa[n] - H[n-1] * z[n-1]) / L[n]
	C[n] = z[n]

	for i in reversed(range(n)):
		C[i] = z[i] - mi[i] * C[i+1]
		B[i] = (A[i+1] - A[i]) / H[i] - H[i] * (C[i+1] + 2 * C[i]) / 3
		D[i] = (C[i+1] - C[i]) / (3 * H[i])
	

	T = np.arange(0., 40., 0.01)
	y = np.arange(0., 40., 0.01)

	
	for j in range(n):
		i = 0
		T = np.arange(0.+10*i, 10+10*i, 0.01)
		for t in T:
			y[i] = A[0] + B[0]*(t-X[0])+C[0]*(t-X[0])**2+D[0]*(t-X[0])**3
			i = i+1	

	T = np.arange(0., 40., 0.01)

	plt.plot(T,y)
	plt.show()
	

		 
		


		 
