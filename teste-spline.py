import numpy as np
from spline import * 

x = [0,1,2,3]
y = [1,np.exp(1),np.exp(2),np.exp(3)]

fpo = 1
fpn = np.exp(3)

spline(x,y,fpo, fpn)
