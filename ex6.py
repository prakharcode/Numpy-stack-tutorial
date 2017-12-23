#plotting cocentric circle using x = r*cos(θ) and y = r*sin(θ)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
A = math.radians(360)*np.random.ranf((100))
A = map(math.degrees,A)
df = pd.DataFrame(A)
df[1] = df.apply(lambda x: 5*math.cos(x[0]), axis=1)
df[2] = df.apply(lambda x: 5*math.sin(x[0]), axis=1)
df[3] = df.apply(lambda x: 10*math.cos(x[0]), axis=1)
df[4] = df.apply(lambda x: 10*math.sin(x[0]), axis=1)
plt.scatter(df[1], df[2])
plt.scatter(df[3], df[4])
plt.axis('equal')
plt.show()
