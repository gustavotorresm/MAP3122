import numpy as np
import matplotlib.pyplot as plt

def spline(X, A, fpo, fpn):
	n = len(X)-1
	H = np.empty((n + 1))
	Alfa = np.empty((n + 1))
	L = np.empty((n + 1))
	mi = np.empty((n + 1))
	z = np.empty((n + 1))
	C = [0] * (n + 1)
	B = [0] * (n + 1)
	D = [0] * (n + 1)

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

	A = A[0:len(A)-1]
	B = B[0:len(B)-1]
	C = C[0:len(C)-1]
	D = D[0:len(D)-1]


	print(A)
	print(B)
	print(C)
	print(D)


#	T = np.arange(1., 3., 0.1)
#	y = np.zeros(20)
#	j = 0
#	for i in range(3):
#		T = np.arange(0.+1*i, 1+1*i, 0.1)
#		print(T)
#		for t in T:		
#			y[j] = A[i] + B[i]*(t-X[i])+C[i]*(t-X[i])**2+D[i]*(t-X[i])**3
#			j = j+1
#	T = np.arange(0., 3., 0.1)
#	plt.plot(T,y)
#	plt.show()
