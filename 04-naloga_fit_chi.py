import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, curve_fit
import scipy

x_i = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 
                6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0])
y_i = np.array([0.317, 0.438, 0.565, 0.561, 0.637, 0.651, 0.635, 0.645, 
                0.609, 0.624, 0.615, 0.572, 0.543, 0.503, 0.503, 0.461, 
                0.425, 0.408, 0.416, 0.367, 0.331])
sigma_i = np.array([0.0155, 0.0172, 0.0160, 0.0154, 0.0142, 0.0165, 0.0144, 
                    0.0189, 0.0196, 0.0138, 0.0179, 0.0153, 0.0157, 0.0193, 
                    0.0107, 0.0109, 0.0102, 0.0183, 0.0178, 0.0187, 0.0198])


def func(x, a, b):
    return a * x * np.exp(b * x)

def chi(params, x_i, y_i, sigma_i):
    a, b = params
    y_pred = func(x_i, a, b)
    return np.sum(((y_i - y_pred) / sigma_i) ** 2)

res_chi2 = minimize(chi, x0=[1, -0.1], args=(x_i, y_i, sigma_i))
a_chi2, b_chi2 = res_chi2.x

popt, pcov = curve_fit(func, x_i, y_i, sigma=sigma_i, absolute_sigma=True)
a_fit, b_fit = popt

x_fit = np.linspace(min(x_i), max(x_i), 100)
y_chi2_fit = func(x_fit, a_chi2, b_chi2)
y_scipy_fit = func(x_fit, a_fit, b_fit)

fitParams, fitCovariances = scipy.optimize.curve_fit(func, x_i, y_i, p0 = [0.7, 0.2], sigma=sigma_i)
sigma_a, sigma_b = np.sqrt(np.diag(fitCovariances))

plt.figure(figsize=(8, 5))
plt.errorbar(x_i, y_i, yerr=sigma_i, fmt='ro', label="Meritve", capsize=3)
plt.plot(x_fit, y_chi2_fit, 'g-', label=r"Chi^2 Fit")
plt.plot(x_fit, y_scipy_fit, 'm:', label="Scipy Fit")

plt.plot(x_i, func(x_i, fitParams[0], fitParams[1]),
     x_i, func(x_i, fitParams[0] + np.sqrt(fitCovariances[0,0]), fitParams[1] - np.sqrt(fitCovariances[1,1])),
     x_i, func(x_i, fitParams[0] - np.sqrt(fitCovariances[0,0]), fitParams[1] + np.sqrt(fitCovariances[1,1])))



plt.xlabel("x")
plt.ylabel("y")
plt.title("Prilagajanje funkcije na podatke")
plt.legend()
plt.grid(True)
plt.show()


print("Optimalni parametri:")
print(f"Chi^2 metoda:      a = {a_chi2:.4f}, b = {b_chi2:.4f}")
print(f"Scipy curve_fit:   a = {a_fit:.4f}, b = {b_fit:.4f}")

def model_3params(x, a, b, c=0):  
    return a * x * np.exp(b * x) + c

def chi_squared_3params(params, x_i, y_i, sigma_i):
    a, b = params  
    y_pred = model_3params(x_i, a, b, c=0)  
    return np.sum(((y_i - y_pred) / sigma_i) ** 2)


a_vals = np.linspace(a_chi2 - 1, a_chi2 + 1, 100)
b_vals = np.linspace(b_chi2 - 1, b_chi2 + 1, 100)

chi2_grid = np.zeros((len(a_vals), len(b_vals)))


for j, a_test in enumerate(a_vals):
    for i, b_test in enumerate(b_vals):
        params_test = [a_test, b_test]
        chi2_grid[j, i] = chi_squared_3params(params_test, x_i, y_i, sigma_i)

plt.figure(figsize=(7, 5))
plt.title(r"$\chi^2$ pri c = 0")
plt.xlabel("a")
plt.ylabel("b")

extent = [a_vals[0], a_vals[-1], b_vals[0], b_vals[-1]]
plt.imshow(chi2_grid, origin='lower', extent=extent, aspect='auto', norm='log')

cbar = plt.colorbar()
cbar.set_label(r'$\chi^2$')

plt.plot([a_chi2], [b_chi2], 'r*', ms=12, label="Best Fit (a, b)", markerfacecolor='none')
plt.legend()
plt.show()

