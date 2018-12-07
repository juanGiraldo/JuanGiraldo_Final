import numpy as np
import matplotlib.pyplot as plt

d1=np.genfromtxt('proceso_1')
d2=np.genfromtxt('proceso_2')
d3=np.genfromtxt('proceso_3')
d4=np.genfromtxt('proceso_4')
d5=np.genfromtxt('proceso_5')
d6=np.genfromtxt('proceso_6')
d7=np.genfromtxt('proceso_7')
d8=np.genfromtxt('proceso_8')

import matplotlib.mlab as mlab
import math

mu = 0
variance = 1
sigma = math.sqrt(variance)
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)

bins = np.linspace(-3, 3, 100)
plt.hist(d1, bins, alpha=0.5, label='proceso_1')
plt.hist(d2, bins, alpha=0.5, label='proceso_2')
plt.hist(d3, bins, alpha=0.5, label='proceso_3')
plt.hist(d4, bins, alpha=0.5, label='proceso_4')
plt.hist(d5, bins, alpha=0.5, label='proceso_5')
plt.hist(d6, bins, alpha=0.5, label='proceso_6')
plt.hist(d7, bins, alpha=0.5, label='proceso_7')
plt.hist(d8, bins, alpha=0.5, label='proceso_8')

plt.plot(x,mlab.normpdf(x, mu, sigma))
plt.plot()
plt.legend(loc='upper right')
plt.savefig('markov_chains.pdf')