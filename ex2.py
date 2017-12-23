'''
Central Limit Theorem
If Y is a sum of infinte X where X is a random variable from any distribution
Then distribution of Y becomes Gaussian.
Here we draw 1000*1000 matrix of random number and take sum row wise hence it
gives 1000 points which we assume are from some distribution Y we plot these
points to enquire the distribution and we get a bell shaped curve indeed.
'''
import numpy as  np
import matplotlib.pyplot as plt

Y = map(np.sum,np.random.ranf((1000,1000))) #continuos uniform dist
Y_norm = map(np.sum,np.random.randn(1000,1000))

plt.hist(Y, bins=100)
plt.hist(Y_norm, bins=100)
plt.show()
