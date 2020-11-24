import numpy as np
import matplotlib.pyplot as plt
import timeit
# Use simple methods as long as they work, whne they dont work anymore, then
# its time to be clever # Use the hammer as long as it works
# monte carlo algorithm for approximation of pi


def calc_pi(r,r_gen, ins):
    A_box = r_gen.flatten().size
    A_circle = 4 * r * r * (r_gen[ins].flatten().size / A_box)
    pi = A_circle / r**2
    return pi

def pi(N, n, r, plot):

    a = np.random.uniform(- 0.5,0.5, size = (N,n)) * r * 2
    b = np.random.uniform(- 0.5,0.5, size = (N,n)) * r * 2

    r_sim = np.sqrt((a * a + b * b))

    inside = r_sim < r
    outside = r_sim > r
    pi = calc_pi(r,r_sim,inside)
    print(pi)
    if plot:
        fig, ax = plt.subplots(figsize = (10,10))
        Villads_siger_jeg_skal_kalde_den_cirkel = plt.Circle((0,0),r, fill = False)
        plt.scatter(a[inside],b[inside], color = 'r', s = 10)
        plt.scatter(a[outside],b[outside], color = 'b', s = 10)
        ax.add_artist(Villads_siger_jeg_skal_kalde_den_cirkel)
        ax.grid(True)
        ax.set_title(fr'$\pi$ ≈ {pi}', fontsize = '18')
        plt.show()

pi(1000,1000,5.2,True)
