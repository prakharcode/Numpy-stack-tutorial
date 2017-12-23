#plotting mean of every image in MNIST dataser

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('train.csv')
mean_number = map(lambda x: df[ df.label == x].mean(axis=0).as_matrix()[1:].reshape(28,28), [i for i in range(10)])

for i in range(10):
    plt.imshow(255-mean_number[i],cmap='gray')
    plt.show()
