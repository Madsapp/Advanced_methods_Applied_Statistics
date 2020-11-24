
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import interpolate
# interpolation are done by eye, keep added and remove smoothness until it looks good
# splining is the duct tape of statistics # last resort

path = %pwd
data = np.loadtxt(path + '/Advanced Applied Stat/Lectures/DustLog_forClass.dat.txt')
spline_cubic = np.loadtxt(path + '/Advanced Applied Stat/Lectures/SplineCubic.txt')


x = np.linspace(np.min(data[:,0]),np.max(data[:,0]),10000)
xlin = np.linspace(np.min(data[:,0]),np.max(data[:,0]),data[:,0].size)
spline = sp.interpolate.CubicSpline(data[:,0],data[:,1])

data[:,0].
plt.figure(figsize = (18,10))
plt.scatter(data[:,0],data[:,1])
plt.plot(data[:,0],np.interp(xlin,data[:,0],data[:,1]), 'g')
plt.plot(x[1800:2000],spline(x)[1800:2000],'r')
plt.grid(True)

# idx = spline_cubic[:,1] < 0.01
# idx2 = spline_cubic[:,0] > 1e-5
# spline_cubic[idx][1:]

xlin2 = np.linspace(np.min(spline_cubic[:,0]),np.max(spline_cubic[:,0]),10000)
spline2 = sp.interpolate.CubicSpline(spline_cubic[:,0],spline_cubic[:,1] )

norm_sp = spline_cubic[:,1]/ np.sum(spline_cubic[:,1])

plt.scatter(spline_cubic[:,0][4:],spline_cubic[:,1][4:])
plt.plot(xlin2[:],spline2(xlin2)[:])
