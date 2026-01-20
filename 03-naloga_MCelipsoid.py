import numpy as np
import matplotlib.pyplot as plt
import timeit
from scipy.integrate import quad
import numpy as np
import scipy.integrate as scint
seed_value = 67891
import os
os.environ['PYTHONHASHSEED'] = str(seed_value)
import random
random.seed(seed_value)
np.random.seed(seed_value)


N = 10000

#parametri elipsoide
a = 2  
b = 0.5  
c = 1



# Define the ellipsoid function
def elipsoid(x, y, z):
    a = 2  
    b = 0.5  
    c = 1
    return (x / a) ** 2 + (y / b) ** 2 + (z / c) ** 2

# Define the limits for integration
x_lower = -a
x_upper = a
y_lower = lambda x: -np.sqrt(1 - (x**2 / a**2))*b
y_upper = lambda x: np.sqrt(1 - (x**2 / a**2))*b
z_lower = lambda x, y: -c*np.sqrt(1 - (x**2 / (a ** 2)) - (y**2 / (b ** 2)))
z_upper = lambda x, y: c*np.sqrt(1 - (x**2 / (a ** 2)) - (y**2 / (b ** 2)))

# Perform the integration
result, error = scint.tplquad(elipsoid, x_lower, x_upper, y_lower, y_upper, z_lower, z_upper)


def monte_carlo_integration(N):
    r1 = np.random.random_sample(N) * 4.0 - 2.0  #[-2, 2]
    r3 = np.random.random_sample(N) * 2.0 - 1.0  #[-1, 1]
    r2 = np.random.random_sample(N) * 1.0 - 0.5  #[-0.5, 0.5]

    x = r1
    y = r2
    z = r3
    
    #preverimo ali so točke v notranjosti
    notr = ((x / a) ** 2 + (y / b) ** 2 + (z / c) ** 2) <= 1

    #če so v notranjosti jih seštejemo
    success = np.sum(notr.astype(int))
    integral = success * (2*a * 2*b * 2*c) / N 

    
    notrT = np.array([x[notr], y[notr], z[notr]])
    zunajT = np.array([x[~notr], y[~notr], z[~notr]])

    return integral, notrT, zunajT

integral, notr, zunaj = monte_carlo_integration(N) 



print("število točk: {}, Seed value: {}".format(N, seed_value))
print("MC integracija = {}".format(integral))
print("Relativna napaka: {}".format(1. - integral / (2*a * 2*b * 2*c)))
print("Rezultat vgrajene integracije(scint.tplquad):", result)
print('razlika med MC integracija in vgrajeno:', integral - result)
print("___________________________________________________________________")

# Plotting
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')


# u = np.linspace(0, 2 * np.pi, 100)
# v = np.linspace(0, np.pi, 100)
# x = 2 * np.outer(np.cos(u), np.sin(v))
# y = 1 * np.outer(np.sin(u), np.sin(v))
# z = 0.5 * np.outer(np.ones(np.size(u)), np.cos(v))
# ax.plot_surface(x, y, z, color='b', alpha=0.3)

# # Plot points
# ax.scatter(notr[0], notr[1], notr[2], c='r', marker='o', label='Inside')
# ax.scatter(zunaj[0], zunaj[1], zunaj[2], c='g', marker='^', label='Outside')

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('Monte Carlo integracija elipsoide')
# ax.legend()

# plt.show()

# loop = 0
# start = timeit.default_timer()
# while loop < 100:
#     monte_carlo_integration(N)
#     loop += 1
# stop = timeit.default_timer()
# potrebenCas2 = stop - start
# print('št. iteracij:', loop)
# print('Monte Carlo integracija elipsoide:', potrebenCas2, 's')

 

# def perform_integration():
#     x_lower = -a
#     x_upper = a
#     y_lower = lambda x: -np.sqrt(1 - (x**2 / a**2))*b
#     y_upper = lambda x: np.sqrt(1 - (x**2 / a**2))*b
#     z_lower = lambda x, y: -c*np.sqrt(1 - (x**2 / (a ** 2)) - (y**2 / (b ** 2)))
#     z_upper = lambda x, y: c*np.sqrt(1 - (x**2 / (a ** 2)) - (y**2 / (b ** 2)))
#     result = scint.tplquad(elipsoid, x_lower, x_upper, y_lower, y_upper, z_lower, z_upper)
#     return result
# loop = 0
# start_time = timeit.default_timer()
# while loop < 100:
#     perform_integration()
#     loop += 1
# stop_time = timeit.default_timer()
# potrebenCas1 = stop_time - start_time

# print('vgrajena integracija(scint.tplquad):', potrebenCas1, 's')




# #REZULTATI

# število točk: 1000, Seed value: 12345
# MC integracija = 4.288
# Relativna napaka: 0.46399999999999997
# Rezultat vgrajene integracije(scint.tplquad): 4.398229715025531
# razlika med MC integracija in vgrajeno: -0.1102297150255307
# ___________________________________________________________________
# št. iteracij: 100
# Monte Carlo integracija elipsoide: 0.03578560001915321 s
# vgrajena integracija(scint.tplquad): 17.77924040000653 s

# število točk: 1000, Seed value: 67891
# MC integracija = 3.928
# Relativna napaka: 0.509
# Rezultat vgrajene integracije(scint.tplquad): 4.398229715025531
# razlika med MC integracija in vgrajeno: -0.470229715025531
# ___________________________________________________________________
# št. iteracij: 100
# Monte Carlo integracija elipsoide: 0.06097090005641803 s
# vgrajena integracija(scint.tplquad): 17.64785489998758 s



#_________________________________________________________________________________


# število točk: 10000, Seed value: 12312
# MC integracija = 4.2104
# Relativna napaka: 0.4737
# Rezultat vgrajene integracije(scint.tplquad): 4.398229715025531
# razlika med MC integracija in vgrajeno: -0.18782971502553103

# število točk: 10000, Seed value: 67891
# MC integracija = 4.1504
# Relativna napaka: 0.48119999999999996
# Rezultat vgrajene integracije(scint.tplquad): 4.398229715025531
# razlika med MC integracija in vgrajeno: -0.24782971502553064


# _________________________________________________________________