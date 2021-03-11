import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier


path = %pwd

bcg_test = np.loadtxt(path +'/Advanced Applied Stat/Lectures/BDT_background_test.txt')
bcg_trained = np.loadtxt(path +'/Advanced Applied Stat/Lectures/BDT_background_train.txt')
sig_test = np.loadtxt(path +'/Advanced Applied Stat/Lectures/BDT_signal_test.txt')
sig_trained = np.loadtxt(path +'/Advanced Applied Stat/Lectures/BDT_signal_train.txt')

fig, ax = plt.subplots(ncols = 3, figsize =(12,8))
ax[0].hist(bcg_test[:,2], bins =100);
ax[0].hist(sig_test[:,2], bins =100);
ax[1].hist(bcg_test[:,1], bins =50);
ax[1].hist(sig_test[:,1], bins =50);
ax[2].hist(bcg_test[:,0], bins =100);
ax[2].hist(sig_test[:,0], bins =100);

fig1, ax1 = plt.subplots(ncols = 3,nrows = 2,figsize =(12,8))
ax1[0,0].scatter(sig_trained[:,0],sig_trained[:,1], label = 'trained XY')
ax1[0,0].scatter(bcg_trained[:,0],bcg_trained[:,1], label = 'trained XY')
ax1[0,0].set_title('XY-test')
ax1[0,1].scatter(sig_trained[:,1],sig_trained[:,2], label = 'trained YZ')
ax1[0,1].scatter(bcg_trained[:,1],bcg_trained[:,2], label = 'trained YZ')
ax1[0,1].set_title('YZ-test')
ax1[0,2].scatter(sig_trained[:,0],sig_trained[:,2], label = 'trained XZ')
ax1[0,2].scatter(bcg_trained[:,0],bcg_trained[:,2], label = 'trained XZ')
ax1[0,2].set_title('XZ-test')
ax1[1,0].scatter(sig_test[:,0],sig_test[:,1], label = 'test XY')
ax1[1,0].scatter(bcg_test[:,0],bcg_test[:,1], label = 'test XY')
ax1[1,0].set_title('XY-test')
ax1[1,1].scatter(sig_test[:,1],sig_test[:,2], label = 'test YZ')
ax1[1,1].scatter(bcg_test[:,1],bcg_test[:,2], label = 'test YZ')
ax1[1,1].set_title('YZ-test')
ax1[1,2].scatter(sig_test[:,0],sig_test[:,2], label = 'test XZ')
ax1[1,2].scatter(bcg_test[:,0],bcg_test[:,2], label = 'test XZ')
ax1[1,2].set_title('XZ-test')

Boost = AdaBoostClassifier(n_estimators=90, random_state=0)

xtreme_Test = np.concatenate((sig_test,bcg_test))
xtremely_buff = np.concatenate((sig_trained,bcg_trained))


binary = np.hstack((np.zeros((2000)),np.ones((2000))))
binarydos = np.hstack((np.zeros((4000)),np.ones((4000))))



Boost.fit(xtremely_buff,binarydos)
Boost.score(xtreme_Test,binary)
Guess = Boost.predict(xtreme_Test)

two_homies = Boost.decision_function(xtreme_Test)



plt.hist(two_homies[binary == 0], bins = 60);
plt.hist(two_homies[binary == 1], bins = 60);
