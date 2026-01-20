import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root

import timeit
##Funkcije 

def f(x):
    ##nicle
    #return np.cos(4*x) + x
    #return x**3-3*x**2+x+1
    return -x**3+x**2+x

def odvod(x):
    #return -4*np.sin(4*x) - 1
    #return 3*x**2-2*x+1
    return -3*x**2+2*x+1
meja_a = -1 
meja_b = 2
plrange = np.linspace(meja_a, meja_b, 100)
newton_parameter = 0.5


##bisekcija

def bisekcija(f, a, b):
  a = a * 1.0 
  b = b * 1.0
  n = 1 
  epsilon = 10 ** -15 
  max_korakov = 420 
  tocke = np.array([a,b])
  ret = False 
  while  n <= max_korakov:
    ret = ( a + b ) / 2.0  
    tocke = np.append(tocke,ret)
    if abs(b-a) < epsilon:
        return ret,n, tocke
    if abs(f(ret)) < epsilon: 
      return ret, n, tocke
    else:
      n = n + 1 
      if f(ret) * f(a) > 0: 
        a=ret
      else:
        b=ret

  return ret, n, tocke


foo = bisekcija(f, meja_a, meja_b)
print ("bisekcija :", foo[1], "\t<<koraki || rezultat>>", foo[0])


##regula falsi
def rFalsi(f, a ,b):
  a = a * 1.0
  b = b * 1.0
  epsilon = 10 ** -2 
  max_korakov = 420 
  n=1
  ret = False
  tocke=np.array([a,b])
  while n<=max_korakov:
    ret = b - (b - a) / (f(b) - f(a)) * f(b)  
    tocke=np.append(tocke,ret)
    if abs(ret-b) < epsilon:
      return ret, n, tocke
    else:
        if f(a) * f(ret) < 0:
            b = ret
        else:
            a = ret
    n = n + 1 
  return ret, n, tocke

foo = rFalsi(f,meja_a,meja_b)
print ("regule falsi    :",foo[1], "\t<<koraki || rezultat>>", foo[0])  


def newtonraphson(f, odvod, a):
  a = a * 1.0
  epsilon = 10 ** -15
  max_korakov = 420 
  n = 1
  ret = False
  tocke = np.array([a])
  while n <= max_korakov:
    ret = a - ( f(a) / odvod(a) )
    
    tocke=np.append(tocke, ret)
    if abs(ret - a) < epsilon:
      return ret, n, tocke
    else:
      a = ret
    n = n + 1 
  return ret, n, tocke
foo = newtonraphson(f, odvod, newton_parameter)
print ("newton    :",foo[1], "\t<<koraki || rezultat>>", foo[0])  #Newton Rhapsonova metoda, ničla in št. korakov


##Numpy.roots- polinomi
#koeficenti = [ 1, -3, 1, 1]
koeficenti = [ -1, 1, 1, 0]
nicle = np.roots(koeficenti)
print('numpy ničle:', nicle)


##SciPy
x0 = float(input('vstavi predvidevan položaj nićle:'))
rezultat = root(f, x0)
print('ničle je na mestu:', rezultat.x)






loop = 0
start = timeit.default_timer()
while loop < 1000:
    rFalsi(f, -0.5, 2.5)
    loop += 1

stop = timeit.default_timer()
potrebenCas0 = stop - start
print('Regula falsi: ', potrebenCas0, 's')
print('________________________________________________________________')

loop = 0
start = timeit.default_timer()
while loop < 1000:
    bisekcija(f, -0.5, 2.5)
    loop += 1

stop = timeit.default_timer()
potrebenCas1 = stop - start
print('Bisekcije: ', potrebenCas1, 's')
print('________________________________________________________________')

loop = 0
start = timeit.default_timer()
while loop < 1000:
    newtonraphson(f, odvod, 2.5)
    loop += 1

stop = timeit.default_timer()
potrebenCas2 = stop - start
print('Newton Raphson : ', potrebenCas2, 's')
print('________________________________________________________________')