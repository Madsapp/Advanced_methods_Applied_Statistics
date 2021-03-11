import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import stats

# Husk buster dot vector instead of for loops # google
## Lecture 4
# include 90 of distribution and set that as uncertaincy
# more data will save you, with the likelihood overtaking your distribution
K = 100
k = 10
n = 60

def factorial(n):
    fact = 1
    for num in range(2,n + 1):
        fact *= num
    return fact

def binom(n,k):
    frac = factorial(n) / (factorial(k) * factorial(n - k))
    return frac


def hyper_binom(N,n,K,k):
    arr = np.zeros(N.size)
    lower ,higher = np.min(N), np.max(N)
    for i in range(lower,higher):
        arr[i-lower] = (binom(K,k) * binom((i-K), (n-k))) / binom(i,n)
    return arr
#
# N = np.zeros((3000))
# N2 = np.zeros((3000))
# for i in range(3000):
#     N[i] = hyper_binom(i + K,n,K,k)
#     N2[i] = N[i] / (i + K)
#
# N3 = np.zeros((3000))
# N4 = np.zeros((3000))
#
# for i in range(3000):
#     N3[i] = hyper_binom(i + K, n, K, 15)
#     N4[i] = N3[i]/ (i + K)
#
# norm_const = np.sum(N)
# norm_const2 = np.sum(N2)
# norm_const3 = np.sum(N3)
# norm_const4 = np.sum(N4)
#
# plt.figure(figsize=(10,10))
# plt.plot(np.arange(3000),N/norm_const)
# plt.plot(np.arange(3000),N2/norm_const2)
# plt.plot(np.arange(3000),N3/norm_const3)
# plt.plot(np.arange(3000),N4/norm_const4)
# plt.grid(True)

# Monte monte
#

vol = 5000
vol_std = 300
fish_dens = 10
fish_std = 1
fish_func = vol / fish_dens

def error_prob(var1,std1,var2,std2):
    error = (1 / var2)**2 * std1**2 + (var1 / var2**2)**2 * std2**2
    return np.sqrt(error)

error_prob(vol, vol_std,fish_dens,fish_std)



def stdfish(size):
    rang = np.linspace(3000,8000,size)
    rang2 = np.linspace(0,20,size)
    vol_distr = gauss(rang,5000,300,True)
    fish_distr = gauss(rang2,10,10,True)
    return (vol_distr / fish_distr) / np.sum((vol_distr / fish_distr))

fish = stdfish(1000)


plt.plot(fish)
plt.axvline(400)

monte_carlo(1000)


def monte_carlo(size):
    xx = np.linspace(0,800,size)

    norm1 = np.random.uniform(0, 800,   size = size)
    norm2 = np.random.uniform(0, 0.004672278775877802,       size = size)

    # norm_const = np.sum(stdfish(xx))

    ab = stdfish(size)

    inside = ab > norm2
    outside = ab < norm2


    plt.figure(figsize = (10,10))
    plt.scatter(norm1[inside],norm2[inside], color = 'b')
    plt.scatter(norm1[outside],norm2[outside], color = 'r')
    plt.plot(xx,ab)
    return ab[inside]


# This does something
list = []
for i in range(100):
    vol_new = np.random.normal(5000,300,20000)
    fishD_new = np.random.normal(10,1,20000)

    Sig = np.mean(error_prob(vol_new,vol_std,fishD_new,fish_std))
    list.append(Sig)

np.mean(list)
##

"""
Try monte_carlo for the two distributions multiplied -> likelihood
"""


error = error_prob(vol, vol_std,fish_dens,fish_std)

def gauss(x,mu,sigma,norm):
    if norm:
        norm_const = 1. / (np.sqrt(2 * np.pi * sigma**2))
        dist = np.exp(- ((x - mu) * (x - mu)) / (2 * sigma)**2)
        func = norm_const * dist
    else:
        func = np.exp(- ((x - mu) * (x - mu)) / (2 * sigma)**2)
    return func

x = np.array([1.01,1.3,1.35,1.44])

def log_likelihood_gauss(x,sigma,mu,norm):
    LH = np.log(gauss(x,sigma,mu,norm))
    return LH

error


N_fish = np.arange(100,700)

plt.plot(N_fish,gauss(N_fish,error,fish_func,True))

norm = np.sum(hyper_binom(N_fish,30,50,8))

normc = np.sum(hyper_binom(N_fish,30,50,8))

mult = hyper_binom(N_fish,30,50,8) * gauss(N_fish,error,fish_func,True)

plt.plot(N_fish,hyper_binom(N_fish,30,50,4)/normc,label = 'k = 4')
plt.plot(N_fish,hyper_binom(N_fish,30,50,8)/normc,label = 'k = 8')
plt.plot(N_fish,mult/ np.sum(mult),label = 'gauss')
plt.legend()
