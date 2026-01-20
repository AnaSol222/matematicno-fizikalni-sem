import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import timeit



def gauss(x, mu, sigma):
    return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
mu = 0  # Mean
sigma = 1  # standrdna divijacija



# quad integracija za primerjavo
integral_value_quad, error = quad(gauss, -np.inf, np.inf, args=(mu, sigma))
print("Gaussian integral (quad):", integral_value_quad)




# MC integracija
seed_value = 67891
import os
os.environ['PYTHONHASHSEED'] = str(seed_value)
import random
random.seed(seed_value)
np.random.seed(seed_value)



N = 1000

x_samples_mc = np.random.normal(mu, sigma, N)
y_samples_mc = np.random.uniform(0, 0.4, N)

gaus_vrednosti = gauss(x_samples_mc, mu, sigma)
pod_mc = y_samples_mc <= gaus_vrednosti
success_mc = np.sum(pod_mc/ 0.4)
estimate_mc = 0.4*success_mc/ N 



plt.scatter(x_samples_mc[pod_mc], y_samples_mc[pod_mc], marker='^', c='blue')
plt.scatter(x_samples_mc[~pod_mc], y_samples_mc[~pod_mc], marker='o', c='red')
plt.plot(np.linspace(-2, 2, 1000), gauss(np.linspace(-2, 2, 1000), mu, sigma), color='black', label='Gauss funkcija')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Monte Carlo integracija')
plt.legend()
plt.grid()
plt.show()
print("število točk: {}, Seed: {}, MC integracija: {}, relativna napaka (površina integrala je defenirana preko quad): {}".format(N, seed_value, estimate_mc, np.abs(integral_value_quad - estimate_mc )))
print("točke pod funkcijo:", success_mc)
print("točke nad funkcijo:", N - success_mc)

# Trapezna metoda integracije
def trapezoidal_integration(gauss, a, b, n):
    h = (b - a) / n
    integral = 0.5 * (gauss(a) + gauss(b))
    for i in range(1, n):
        integral += gauss(a + i * h)
    return integral * h

koraki = 1000
spodnja = -5.0
zgornja = 5.0

# # trapezna metoda
# x_values_trap = np.linspace(spodnja, zgornja, koraki)
# gaussian_values_trap = gauss(x_values_trap, mu, sigma)
# integral_value_trap = trapezoidal_integration(lambda x: gauss(x, mu, sigma), spodnja, zgornja, koraki)

# gaussian_values_trap /= integral_value_trap


# plt.plot(x_values_trap, gaussian_values_trap, label='Gaussian Function')
# plt.fill_between(x_values_trap, gaussian_values_trap, color='skyblue', alpha=0.3)
# plt.xlabel('x')
# plt.ylabel('f(x)')
# plt.title('Trapezna metoda')
# plt.axhline(0, color='black', linewidth=0.5)
# plt.axvline(spodnja, color='red', linestyle='--', linewidth=0.5)
# plt.axvline(zgornja, color='red', linestyle='--', linewidth=0.5)

# for i in range(len(x_values_trap) - 1):
#     x1 = x_values_trap[i]
#     x2 = x_values_trap[i + 1]
#     y1 = gaussian_values_trap[i]
#     y2 = gaussian_values_trap[i + 1]
#     plt.fill([x1, x2, x2, x1], [0, 0, y2, y1], color='green', alpha=0.3)

# plt.text(0.5, 0.2, f'Integral: {integral_value_trap:.4f}', bbox=dict(facecolor='white', alpha=0.5))
# plt.legend()
# plt.grid()
# plt.show()

# print("število korakov: {}, trapezna integracija: {}, Primerjava z vgrajeno integracijo pythona: {}".format(koraki, integral_value_trap, integral_value_quad - integral_value_trap ))


# #REZULTATI

# # Gaussian integral (quad): 0.9999999999999997
# število točk: 1000, Seed: 12345, MC integracija: 0.718, relativna napaka (površina integrala je defenirana preko quad): 0.2819999999999997
# točke pod funkcijo: 1795.0
# točke nad funkcijo: 795.0
# število korakov: 1000, trapezna integracija: 0.9999994143527641, Primerjava z vgrajeno integracijo pythona: 5.856472355958431e-07

# število točk: 1000, Seed: 12345, MC integracija: 0.723, relativna napaka (površina integrala je defenirana preko quad): 0.2769999999999997
# točke pod funkcijo: 1807.5
# točke nad funkcijo: -807.5
# število korakov: 1000, trapezna integracija: 0.9999994265729691, Primerjava z vgrajeno integracijo pythona: 5.734270305257638e-07
