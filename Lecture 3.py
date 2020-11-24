import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

# if you forget likelihood, LOOD at slides ! :P
# never write log, be specific log10 or ln
# LLH: log likelihood
# negative LLH finds minima
def gauss(x,sigma,mu):
    norm_const = 1. / (np.sqrt(2 * np.pi * sigma))
    dist = np.exp(- ((x - mu) * (x - mu)) / (2 * sigma))
    return norm_const * dist

x = np.array([1.01,1.3,1.35,1.44])

def log_likelihood_gauss(x,sigma,mu):
    LH = np.sum(np.log(gauss(x,sigma,mu)))
    return LH

rand_points = np.random.normal(0.2,0.1,50)


def MLH_gauss(mul,muu,sigl,sigu,points):
    mu = np.linspace(mul,muu,points)
    sigma = np.linspace(sigl,sigu,points)
    arr = np.zeros((points,points))
    rand_points = np.random.normal(0.2,0.1,points)
    for i in range(points):
        for j in range(points):

            arr[i,j] = log_likelihood_gauss(rand_points,sigma[i],mu[j])
            print(arr[i,j])
    plt.figure(figsize = (10,10))
    plt.imshow(arr, interpolation='none', extent=[mul,muu,sigu,sigl])
    plt.colorbar()
    return arr

MLH_gauss(0,0.3,0.01,1,20)

## Exercise 2

# try to integrate analytically

def E2_LHfunc(x,alpha,beta,size):
    xx = np.linspace(-1,1,size)
    func = 1 + alpha * x + beta * x * x
    norm_const = 2 + 2/3 * beta
    return func / norm_const

def monte_carlo(rang, size):
    xx = np.linspace(-1,1,rang)

    x = np.random.uniform(-1 ,1 ,   size = size)
    y = np.random.uniform( 0 ,0.02 ,size = size)

    norm_const = np.sum(E2_LHfunc(xx,0.5,0.5,size))

    f = np.vstack((x,y))

    ab = E2_LHfunc(f[0],0.5,0.5,size) / norm_const

    inside = ab > f[1]
    outside = ab < f[1]


    # plt.figure(figsize = (10,10))
    # plt.scatter(f[0][inside],f[1][inside], color = 'b')
    # plt.scatter(f[0][outside],f[1][outside], color = 'r')
    # plt.plot(xx,E2_LHfunc(xx,0.5,0.5,size)/norm_const)
    return f[0][inside]


def log_likelihood(x,alpha,beta,size):
    LH = np.sum(np.log(E2_LHfunc(x,alpha,beta,size)))
    return LH

log_likelihood(monte_carlo(100,4000),0.5,0.5,100)

# Class 3 exercise 1 to speed up calculations
def MLH_exer2(alp,bet,points):
    p = monte_carlo(points,4000)
    alpha = np.linspace(0,alp,points)
    beta = np.linspace(0,bet,points)
    arr = np.zeros((points,points))
    for i in range(points):
        for j in range(points):
            arr[i,j] = log_likelihood(p,alpha[i],beta[j],points)


    plt.imshow(arr.T,vmin = -1400, origin = 'lower', extent = [0,alp,0,bet])
    # plt.scatter()
    plt.colorbar()
    return arr

MLH_exer2(1,1,100);
