import numpy as np
import matplotlib.pyplot as plt
import timeit
import scipy.integrate as scint
from scipy.integrate import quad

# Seed value
seed_value = 12312

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED'] = str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)

a = 2  
b = 1   #
a_limit = -2  # Lower limit of integration for x
b_limit = 2 

N = 1000
def elipsa(x, y):
    a = 2
    b = 1
    return (x**2 / a**2 + y**2 / b**2)


result = scint.dblquad(elipsa, -2, 2, lambda x: -np.sqrt(1 - (x**2 / 4)), lambda x: np.sqrt(1 - (x**2 / 4)))

def monte_carlo_integration(N):
    r1 = np.random.random_sample(N) *  4.0 - 2.0  
    r2 = np.random.random_sample(N) * 2.0 - 1.0  
    x = r1
    y = r2
    inside = (x**2 / a**2 + y**2 / b**2) <= 1
    success = np.sum(inside.astype(int))
    estimate = success * (2*a * 2*b) / N 
    return estimate, inside, success, x, y

estimate, inside, success, x, y = monte_carlo_integration(N)

# izračun točk
points_inside = np.sum(inside)

# Print infromacije
print("število točk: {}, Seed value: {}".format(N, seed_value))
print('MC integracija:', estimate)
print("relativna napaka: {}".format(1. - estimate / ( 2*a * 2*b)))
print("točke znotraj elpise:", points_inside)
print("točke zunaj elipse:", N - points_inside)
print("povrsina scipy quad :", result)
print("razlika quad in MC integracija:", result - estimate)
print("___________________________________________________________________")


plt.figure(figsize=(8, 6))

theta = np.linspace(0, 2 * np.pi, 100)
plt.plot(a * np.cos(theta), b * np.sin(theta), color='black', label='Elipsa')  # Plot elipse

# Plot točk znotraj in zunaj elipse
plt.scatter(x[inside], y[inside], marker='o', c='blue', alpha=0.5, label='notri')
plt.scatter(x[~inside], y[~inside], marker='x', c='red', alpha=0.5, label='zunaj')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Monte Carlo integracija')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()


loop = 0
start = timeit.default_timer()
while loop < 1000:
    monte_carlo_integration(N)
    loop += 1
stop = timeit.default_timer()
potrebenCas2 = stop - start
print('2D Monte Carlo intergacija:', potrebenCas2, 's')
loop = 0
start = timeit.default_timer()
def Vintegracija():
    result = scint.dblquad(elipsa, -2, 2, lambda x: -np.sqrt(1 - (x**2 / 4)), lambda x: np.sqrt(1 - (x**2 / 4)))
    return result
while loop < 1000:
    Vintegracija()
    loop += 1
stop = timeit.default_timer()
potrebenCas3 = stop - start
print('Vgrajena integracija(scint.dblquad():', potrebenCas3, 's')

#REZULTATI
# število točk: 1000, Seed value: 12345
# MC integracija: 6.352
# relativna napaka: 0.20599999999999996
# točke znotraj elpise: 794
# točke zunaj elipse: 206
# povrsina scipy quad : 6.675884388878311
# razlika quad in MC integracija: 0.3238843888783105
# 2D Monte Carlo intergacija: 0.3072196000139229 s
# Vgrajena integracija(scint.dblquad(): 9.077082400035579 s

# število točk: 1000, Seed value: 67891
# MC integracija: 6.152
# relativna napaka: 0.23099999999999998
# točke znotraj elpise: 769
# točke zunaj elipse: 231
# povrsina scipy quad : 6.675884388878311
# razlika quad in MC integracija: 0.5238843888783107
# 2D Monte Carlo intergacija: 0.40096709999488667 s
# Vgrajena integracija(scint.dblquad()): 9.103123500011861 s


#____________________________________________________________________________________


# število točk: 10000, Seed value: 12345
# MC integracija: 6.2928
# relativna napaka: 0.21340000000000003
# točke znotraj elpise: 7866
# točke zunaj elipse: 2134
# povrsina scipy quad : 6.675884388878311
# razlika quad in MC integracija: 0.3830843888783111
# ___________________________________________________________________


# število točk: 10000, Seed value: 67891
# MC integracija: 6.2392
# relativna napaka: 0.22009999999999996
# točke znotraj elpise: 7799
# točke zunaj elipse: 2201
# povrsina scipy quad : 6.675884388878311
# razlika quad in MC integracija: 0.4366843888783105
# ___________________________________________________________________

