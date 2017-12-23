#plot a spiral
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
B = abs(np.random.ranf((10000)))
A = math.radians(90)*B
A = map(math.degrees,A)
df = pd.DataFrame(A)
df[1]=B
df[2] = df.apply(lambda x: 5*x[1]*math.cos(x[0]), axis=1)
df[3] = df.apply(lambda x: 5*x[1]*math.sin(x[0]), axis=1)
plt.scatter(df[2], df[3])
plt.axis('equal')
plt.show()
