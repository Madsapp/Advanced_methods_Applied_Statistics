import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from matplotlib.patches import Ellipse
from scipy.optimize import minimize
import timeit
# Lecture 5
# To next:
# - Fix this piece of shit code
# - contour here
# - Find function for doing markov chains online, because youre gonna write your own, but its gonna be slow af
# p-value distributions, if the hypothesis is correct the p-value distribution is flat/uniform
# if Wilks-theorem fails, then it isnt Chi2 distr anymore

# try to integrate analytically

def E2_LHfunc(x,alpha,beta):
    func = 1 + alpha * x + beta * x * x
    return func

def norm_func(array,tup):
    x_min, x_max = np.min(array), np.max(array)
    alpha,beta = tup
    upperlim = (x_max + 1/2 * alpha * x_max**2 + 1/3 * beta * x_max**3)
    lowerlim = (x_min + 1/2 * alpha * x_min**2 + 1/3 * beta * x_min**3)
    return E2_LHfunc(array,alpha,beta) / (upperlim - lowerlim)

def monte_carlo(L,U,rang, size):
    xx = np.linspace(L,U,rang)

    func = norm_func(xx,(0.31578669, 2.15038608))
    norm_const = np.sum(func)

    x = np.random.uniform(L, U,     size = 8000)
    y = np.random.uniform( 0 , np.max(func/norm_const), size = 8000)

    ab = norm_func(x,(0.31578669, 2.15038608)) / norm_const

    inside = ab > y
    outside = ab < y
    l0 = x[inside]
    l0 = np.random.choice(l0, len(l0) - (len(l0) - size), replace=False)
    # plt.figure(figsize = (10,10))
    # plt.scatter(x[inside],y[inside], color = 'b')
    # plt.scatter(x[outside],y[outside], color = 'r')
    # plt.plot(xx,func/norm_const)
    return l0


def log_likelihood(x,tup):
    LH = np.sum(np.log(norm_func(x,tup)))
    return LH


MLH_exer2(-1,1,1,3,100,True);

# Class 3 exercise 1 to speed up calculations
def MLH_exer2(L,U,alp,bet,points, plot):
    p = monte_carlo(L,U,points,2000)
    alpha = np.linspace(0,alp,points)
    beta = np.linspace(0,bet,points)
    arr = np.zeros((points,points))
    for i in range(points):
        for j in range(points):
            arr[i,j] = log_likelihood(p,(alpha[i],beta[j]))

    a,b = np.unravel_index(arr.argmax(),arr.shape)
    if plot:
        plt.figure(figsize = (10,10))
        plt.imshow(arr.T,vmin = np.max(arr)-8, origin = 'lower', extent = [0,alp,0,bet], cmap = 'jet')
        plt.colorbar()
        plt.scatter(alpha[a],beta[b], label = f'Maximum: [{alpha[a]:.2f} {beta[b]:.2f}]', marker = 'x')
        # plt.contour((arr).T, extent = [0,1,0,1], levels = [np.max(arr) - 5.92, np.max(arr) - 3.09,np.max(arr) - 1.15], colors = ('orange','b','k'))
        plt.xlabel(fr' $\alpha$ ')
        plt.ylabel(fr' $\beta$ ')
        plt.legend()
    return arr, alpha[a], beta[b]


MLH_exer2(-1,1,1,1,100,True);

def big_dick_rasterscan(rang,plot,arr_size):

    arrs = np.zeros((arr_size,arr_size))
    bets = np.zeros(rang)
    alps = np.zeros(rang)

    for i in range(rang):
        skrrt = MLH_exer2(-1,1,1,1,arr_size,False)
        arrs += skrrt[0]
        alps[i] = skrrt[1]
        bets[i] = skrrt[2]


    if plot:
        fig, ax = plt.subplots(figsize = (10,10))
        plt.imshow((arrs).T,vmin = np.max((arrs))-50, origin = 'lower', extent = [0,1,0,1], cmap = 'jet')
        plt.colorbar()
        plt.contour((arrs).T, extent = [0,1,0,1], levels = [np.max(arrs) - 5.92, np.max(arrs) - 3.09,np.max(arrs) - 1.15])
        plt.scatter(np.mean(alps),np.mean(bets),label = f'Maximum: [{np.mean(alps):.2f} {np.mean(bets):.2f}]', marker = 'x')
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.legend()
        plt.show()
    return arrs, alps, bets


raster_array, alpha_values, bets_values = big_dick_rasterscan(10, True, 100)

def ab_histograms(L,U,iter, bins):
    alphas = []
    betas = []
    for i in range(iter):
        m, al, be = MLH_exer2(L,U,1,3,100,False)
        alphas.append(al)
        betas.append(be)

    alphas = np.asarray(alphas)
    betas = np.asarray(betas)

    alpha_s = np.std(alphas)/ np.sqrt(alphas.size)
    beta_s = np.std(betas)/ np.sqrt(betas.size)

    fig ,ax = plt.subplots(ncols = 2, figsize = (10,10))
    ax[0].hist(alphas, bins = bins, color = 'b', label = fr'Estimated alpha, $\mu$ = {np.mean(alphas):.3f}, $\sigma$ = {alpha_s:.3f}')
    ax[1].hist(betas, bins = bins, color = 'r', label = fr'Estimated beta, $\mu$ = {np.mean(betas):.3f}, $\sigma$ = {beta_s:.3f}')
    ax[0].grid(True)
    ax[1].grid(True)
    ax[0].legend()
    ax[1].legend()
    plt.show()
ab_histograms(-1,1,100,bins = 50)

def MLH_exer_1params(L,U,alp,bet,points, plot):
    p = monte_carlo(L,U,points,4000)
    alpha = np.ones(points) - 0.5
    beta = np.linspace(0,bet,points)
    arr = np.zeros((points,points))
    for i in range(points):
        for j in range(points):
            arr[i,j] = log_likelihood(p,alpha[i],beta[j])

    a,b = np.unravel_index(arr.argmax(),arr.shape)
    if plot:
        plt.imshow(arr.T,vmin = np.min(arr)-2, origin = 'lower', extent = [0,alp,0,bet])
        plt.colorbar()
        plt.scatter(alpha[a],beta[b], label = 'Maximum', marker = 'x')
        plt.legend()
    return arr, alpha[a], beta[b]


# Class 3 exercise 1 to speed up calculations
def MLH(p,alp,bet,points, plot):
    alpha = np.linspace(0,alp,points)
    beta = np.linspace(0,bet,points)
    arr = np.zeros((points,points))
    for i in range(points):
        for j in range(points):
            arr[i,j] = log_likelihood(p,alpha[i],beta[j])

    a,b = np.unravel_index(arr.argmax(),arr.shape)
    if plot:
        plt.figure(figsize = (10,10))
        plt.imshow(arr.T,vmin = np.max(arr)-10, origin = 'lower', extent = [0,alp,0,bet])
        plt.colorbar()
        plt.scatter(alpha[a],beta[b], label = f'Maximum: [{alpha[a]:.2f} {beta[b]:.2f}]', marker = 'x')
        plt.xlabel(fr' $\alpha$ ')
        plt.ylabel(fr' $\beta$ ')
        plt.legend()
    return arr, alpha[a], beta[b]

# extra = np.loadtxt('/Users/BareMarcP/Desktop/Mads_stuff/Den Gode Kode/Advanced Applied Stat/MLE_Variance_data_2.txt')

# d1 = extra[:,0]
# d2 = extra[:,1]

# how to avoid divide by 0 when it's outside range
# MLH(d1,1,1,100,True);
# MLH(d2,1,1,100,True);

def min_log_likelihood(ab,x):
    LH = np.sum(-np.log(norm_func(x,ab)))
    return LH

def LHfunc(x,alpha,beta,gamma):
    func = 1 + alpha * x + beta * x * x - gamma * x**5
    return func

def norm_func2(array,tup):#mangler integration
    x_min, x_max = np.min(array), np.max(array)
    alpha, beta, gamma = tup
    upperlim = (x_max + 1/2 * alpha * x_max**2 + 1/3 * beta * x_max**3 - 1/6 * gamma * x_max**6)
    lowerlim = (x_min + 1/2 * alpha * x_min**2 + 1/3 * beta * x_min**3 - 1/6 * gamma * x_min**6)
    return LHfunc(array, alpha, beta, gamma) / (upperlim - lowerlim)

def min_log_likelihood2(abg,x):
    LH = np.sum( - np.log(norm_func2(x,abg)))
    return LH



guess = (0.4,0.4)




# lecture 7 stuff
# min_d1 = minimize(min_log_likelihood, guess, args = d1, method = 'Nelder-Mead')
# fit_d1 = norm_func(d1,min_d1.x[0],min_d1.x[1])
#
# min_d2 = minimize(min_log_likelihood, guess, args = d2, method = 'Nelder-Mead')
# fit_d2 = norm_func(d2,min_d2.x[0],min_d2.x[1])
#
#
# plt.figure(figsize = (10,10))
# plt.plot(d1,fit_d1)
# plt.hist(d1, density=True, bins=200);
#
# plt.figure(figsize=(10,10))
# plt.plot(d2,fit_d2)
# plt.hist(d2, density=True)
#
# plt.hist(fit_d1)

from scipy.stats import chisquare
from scipy.stats import chi2
# chisquare(d1,fit_d1, ddof = 2)
#
# chisquare(d2,fit_d2, ddof = 2)

# lecture 7 first real exercise
path = %pwd

data3 = np.loadtxt(path + '/Advanced Applied Stat/Lectures/Lecture8_LLH_Ratio_2_data.txt')
data4 = np.loadtxt(path + '/Advanced Applied Stat/Lectures/Lecture8_LLH_Ratio_2a_data.txt')

d3 =  data3[:,0]
d4 =  data4[:,0]

guess2 = (0.4,0.4,0.4)

def LLH2_pval(data,func1,func2,guess1, guess2):
    null = minimize(func1, guess,  args = data, method = 'Nelder-Mead')
    alt  = minimize(func2, guess2, args = data, method = 'Nelder-Mead')
    null_fit = norm_func(data,null.x)
    alt_fit  = norm_func2(data,alt.x)

    LLH2 = (null.fun - alt.fun) * 2
    pval = chi2.sf(LLH2,1)
    print(null.fun, alt.fun)
    print(f"-2LLH: {LLH2:.4f} \np-val: {pval:.4f}")

LLH2_pval(d3,min_log_likelihood,min_log_likelihood2,guess,guess2)

#
# ## LLH speed trial
# from scipy.stats import norm
#
# def llh(x, mu, sigma):
#     # Sum over the first dimension.
#     # This will be useful when mu and sigma will be arrays.
#     return np.sum(np.log(norm.pdf(x, loc=mu, scale=sigma)), axis=0)
#
# X = np.random.normal(loc = 0.2, scale = 0.1, size = 50)
#
# mu_range = np.linspace(start = 0, stop = 0.5, num = 100) # Uniform range
# sigma_range = np.geomspace(start = 0.01, stop = 1, num = 100) # Log-uniform range since σ > 0
#
# mu, sigma = np.meshgrid(mu_range, sigma_range, indexing = 'ij')
#
# x_3d = X[:,None,None]
#
# mu_3d = mu[None,:,:]
# sigma_3d = sigma[None,:,:]
#
#
# raster_llh = llh(x_3d, mu_3d, sigma_3d)
#
# raster_max_linear_idx = np.nanargmax(raster_llh)
# id_mu, id_sigma = np.unravel_index(raster_max_linear_idx, raster_llh.shape)
#
# raster_max_mu  = mu_range[id_mu]
# raster_max_sigma = sigma_range[id_sigma]
# raster_max_llh = raster_llh[id_mu, id_sigma]
# print('Maximum LLH (raster): {:.3} reached at μ = {:.3} and σ = {:.3}'.
#       format(raster_max_llh, raster_max_mu, raster_max_sigma))
