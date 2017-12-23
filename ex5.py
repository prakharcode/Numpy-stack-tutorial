#XOR

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

A = np.random.randn(10000,2)
df= pd.DataFrame(A)
df.loc[df[0]>1,0]=1
df.loc[df[0]<-1,0]=-1
df.loc[df[1]>1,1]=1
df.loc[df[1]<-1,1]=-1
df[2] = df.apply(lambda x: x[1]*x[0], axis=1)
df.loc[df[2]>0,2]=0
df.loc[df[2]<0,2]=1


plt.scatter(df[0],df[1],c=df[2])
plt.show()
