import numpy as np

a = np.random.normal(size=(25,12))

b = np.tile(np.random.normal(size=(25,))[:,np.newaxis],(1,12))

print(b)
c = np.add(a,b)

print(c)