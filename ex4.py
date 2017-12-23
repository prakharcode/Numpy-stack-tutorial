import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv('train.csv')
mean_number = map(lambda x: df[ df.label == x].mean(axis=0).as_matrix()[1:].reshape(28,28), [i for i in range(10)])
mean_number2by_loop = []
mean_number2by_func=[]
for k in range(len(mean_number)):
    mean_number2by_loop.append(mean_number[k].copy())
    for i in range(len(mean_number[k])):
        for j in range(len(mean_number[k][i])):
            mean_number2by_loop[k][j][27-i]=mean_number[k][i][j]
    mean_number2by_func.append(np.rot90(mean_number[k], k=3))

print np.equal(np.array(mean_number2by_func),np.array(mean_number2by_loop))

for i in range(10):
    plt.imshow(255-mean_number2by_func[i],cmap='gray')
    plt.show()
