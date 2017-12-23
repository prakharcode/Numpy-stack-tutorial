#Finding Eigenvector
import numpy as np
import matplotlib.pyplot as plt
A = [[ .3, .6, .1],
[.5, .2, .3],
[.4, .1, .5]]
A = np.array(A)
v = [1/3., 1/3., 1/3.]
v = np.array(v)
distance_mat = list()
for i in range(1,25):
    v_new = v.dot(A)
    distance_mat.append(np.abs(np.linalg.norm(v_new-v)))
    v = v_new
plt.plot(range(1,25),distance_mat)
plt.xlabel('Iteration')
plt.ylabel('Distance')
plt.show()
