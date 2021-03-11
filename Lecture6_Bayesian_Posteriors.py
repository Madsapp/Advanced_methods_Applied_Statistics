import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.stats as stats
import markovchain
# posterior proportional til prior * likelihood


alpha = 5
beta = 17

exp_val = alpha / (alpha + beta)
var = (alpha * beta) / ((alpha * beta)**2 * (alpha + beta + 1))

tup1 = alpha,beta,np.linspace(0,1,1000)
tup2 = 66,100,np.linspace(0,1,1000)
x = np.linspace(0,1,1000)

def beta_dist2(a,b,x):
    return stats.beta.pdf(x,a,b)

def beta_dist(tup):
    a,b,x = tup
    return stats.beta.pdf(x,a,b)

def binom_likelihood(tup):
    K,N,p = tup
    return stats.binom.pmf(K,N,p)

binom_likelihood(tup2)

def posterior_func(x, prior_func, prior_tup, likelihood_func, likelihood_tup):

    prior_distr = prior_func(prior_tup)
    likelihood_distr = likelihood_func(likelihood_tup)

    prior = prior_distr / np.sum(prior_distr)
    likelihood = (likelihood_distr) / np.sum(likelihood_distr)
    posterior = (prior * likelihood)/ np.sum(prior * likelihood)
    plt.figure(figsize = (12,8))
    plt.plot(x,prior, label = 'prior')
    plt.plot(x,likelihood, label = 'likelihood')
    plt.plot(x,posterior, label = 'posterior')
    plt.grid()
    plt.legend()
    return prior, likelihood, posterior

prior, likelihood, posterior = posterior_func(x,beta_dist,tup1,binom_likelihood,tup2)



def gauss(x,sigma,Xt):
    norm_const = 1. / (sigma * (np.sqrt(2 * np.pi)))
    dist = np.exp(- ((x - Xt/2) * (x - Xt/2)) / (2 * sigma)**2)
    return norm_const * dist


def markov_simple(start):
    steps = np.zeros((100,len(start))).T
    for j in range(len(start)):
        x = start[j]
        for i in range(100):
            Xt = np.random.normal(x/2,1)
            x = Xt
            steps[j,i] = x

        plt.plot(steps[j])
    return steps

s = (100,-27)

markov_simple(s);


def markov_simple2(start):
    steps = [0.5]
    for j in range(start-1):
        Xt  = steps[-1] + np.random.normal(0,0.3)
        likelihood1 = stats.binom.pmf(66,100,Xt)
        prior1 = beta_dist2(alpha, beta, Xt)
        likelihood2 = stats.binom.pmf(66,100,steps[-1])
        prior2 = beta_dist2(alpha, beta, steps[-1])
        r = ((likelihood1 * prior1) / (likelihood2 * prior2))
        if r > 1:
            steps.append(Xt)
        elif r < 1:
            u = np.random.uniform()
            if r > u:
                steps.append(Xt)
            else:
                steps.append(steps[-1])
    return steps

hist = markov_simple2(5000)

plt.hist(hist, density=True)
plt.plot(x,posterior*1000, label = 'posterior')


import pymcmcstat
from pymcmcstat.MCMC import MCMC
