import numpy as np
from spline import * 

def foo(x):
	return 5 + 8 * x - 2 * x**2 + x**3

x = [0,1,2,3]
y = [foo(0), foo(1), foo(2), foo(3)]

fpo = 8
fpn = 8 - 4 * 3 + 3 * 3**2

spline(x,y,fpo, fpn)
