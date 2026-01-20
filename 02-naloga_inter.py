import numpy as np
from pylab import *
from scipy.integrate import quad
import scipy.integrate as scint
import matplotlib.pyplot as plt
import timeit


def f(x):
    return 1-x**2
    #return x**3*(np.exp(-x))


def simpson(f, a, b, n): 
    h = (b - a) / n  
    s = f(a) + f(b) 

    for i in range(1, n, 2):
        s += 4 * f(a + i * h)
    for i in range(2, n-1, 2):
        s += 2 * f(a + i * h)

    return s * h / 3.0


def trapezoidal(f, a, b, n): 
    h = float(b - a) / n
    s = 0.0
    s += f(a)/2.0
    for i in range(1, n):
        s += f(a + i*h)
    s += f(b)/2.0
    return s * h
if __name__ == '__main__':

    import math


meja_a = -1.  # spodnja meja
meja_b = 1.  # zgornja meja
# meja_a = 0
# meja_b = 50#np.inf
koraki = 1000
rezQuad, _ = scint.quadrature(f, meja_a, meja_b)
rezTrap = trapezoidal(f, meja_a, meja_b, koraki)
rezSim = simpson(f, meja_a, meja_b, koraki)
print("\n")
print("povrsina trapl :", trapezoidal(f, meja_a, meja_b, koraki), '| odstopanje: ', abs(rezQuad - rezTrap))
print("povrsina simps :", simpson(f, meja_a, meja_b, koraki), '| odstopanje: ', abs(rezQuad - rezSim))
print("povrsina scipy quadrature :", scint.quadrature(f, meja_a, meja_b))
print("povrsina scipy quad :", scint.quad(f, meja_a, meja_b))
print("povrsina scipy romberg :", scint.romberg(f, meja_a, meja_b))




if __name__ == '__main__':
    plrange = np.linspace(meja_a, meja_b, koraki * 10)
    plpoints = np.linspace(meja_a, meja_b, koraki)
    
    p = plt.plot(plrange, f(plrange), 'b')  
    plt.grid()
    
    q = plt.plot(plpoints, f(plpoints), 'r.')  
    plt.legend(('funkcija', 'vzorčenje'), loc='lower right')
    
    
    for i in range(koraki - 1):
        x = np.linspace(plpoints[i], plpoints[i+1], 10)
        y = f(plpoints[i]) * np.ones_like(x)
        plt.plot(x, y, 'g--')  
        plt.plot([plpoints[i], plpoints[i]], [0, f(plpoints[i])], 'k--')  
          
        
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title("Integracija")
    plt.show() 
loop = 0
start = timeit.default_timer()
while loop < 1000:
    simpson(f, -1, 1, 100)
    loop += 1

stop = timeit.default_timer()
potrebenCas1 = stop - start
print('simpsonova metoda: ', potrebenCas1, 's')
print('________________________________________________________________')

loop = 0
start = timeit.default_timer()
while loop < 1000:
    trapezoidal(f, -1, 1, 100)
    loop += 1

stop = timeit.default_timer()
potrebenCas2 = stop - start
print('trpezna metoda : ', potrebenCas2, 's')
print('________________________________________________________________')

##prva funkcija rezultati:
# povrsina trapl : 1.3333319999999993 | odstopanje:  1.333333333519704e-06
# povrsina simps : 1.333333333333334 | odstopanje:  1.1102230246251565e-15
# povrsina scipy quadrature : (1.3333333333333328, 6.661338147750939e-16)
# povrsina scipy quad : (1.3333333333333335, 1.4802973661668755e-14)
# povrsina scipy romberg : 1.3333333333333333


##druga funkcija rezultati:
# povrsina trapl : 6.000000052052336 | odstopanje:  4.910233286636867e-08
# povrsina simps : 5.999999792286135 | odstopanje:  2.1066386768353595e-07
# povrsina scipy quadrature : (6.000000002950003, 2.6498533145513647e-08)
# povrsina scipy quad : (6.0, 1.6451046531820913e-13)
# povrsina scipy romberg : 5.999999999999987


