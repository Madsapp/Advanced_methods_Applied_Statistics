import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from scipy.stats import norm

# The p-value never tells you if your data and model agrees
# the power of the test
# Kernel density estimator (KDE), non parametric
# Crab nebula is the standard for gamma ray telescopes.
# People thought it was constant in emittance, turned out it wasn't
# THE KERNEL IS ALWAYS NORMALIZED 
# THIS SHIT NEEDS TO BE NORMALIZED!!!!!


data = np.array([1,2,5,6,12,15,16,16,22,22,22,23])


fig, ax = plt.subplots()
bins = np.arange(1, 23)
ax.plot(data, np.full_like(data, -0.5), '|k',
        markeredgewidth = 1)
for count, edge in zip(*np.histogram(data, bins)):
    for i in range(count):
        ax.add_patch(plt.Rectangle((edge, i), 1.5, 1.5,
                                   alpha=0.5))
ax.set_xlim(0, 24)
ax.set_ylim(-0.2, 3)



x_d = np.linspace(0, 25, 2000)
density1 = sum((abs(xi - x_d) < 1) for xi in data)

plt.fill_between(x_d, density1, alpha = 0.5)
plt.plot(data, np.full_like(data, -0.5), '|k', markeredgewidth=1)



list = [6,10.1,20.499,20.501]

x_d = np.linspace(-5, 30, 1000)
density = sum(norm(loc = xi, scale = 3).pdf(x_d) for xi in data)
# for i in range(len(data)):
#     kern = sum(norm(loc = data[i], scale = 3).pdf(x_d) for xi in data)
#     plt.plot(x_d,kern/(np.sum(kern)), 'r--')
plt.fill_between(x_d,density/np.sum(density), alpha = 0.5)
plt.plot(data, np.full_like(data, -0.5), '|k', markeredgewidth = 1)
plt.axis([-5, 30, 0, np.max(kern)/np.sum(kern)]);
