import math
import numpy as np
from matplotlib import pyplot as plt

with open('out_matxr.txt') as f:
    lines = f.read().replace("/n", "").split(',')

lines = list(map(float, lines))

size = int(math.sqrt(len(lines)))
if size * size != len(lines):
    raise ValueError("Matrix is not square")

array = np.array(lines)
matrix = array.reshape(size, size)

if size < 10:
    print(matrix)

plt.imshow(matrix, interpolation='nearest', cmap='gist_heat')
plt.show()
