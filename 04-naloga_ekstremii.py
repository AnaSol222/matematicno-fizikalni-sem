import math
import matplotlib.pyplot as plt
import numpy as np
import scipy


invphi = (math.sqrt(5) - 1) / 2  # 1 / phi
invphi2 = (3 - math.sqrt(5)) / 2  # 1 / phi^2

def zlatirez(f, a, c, tol=1e-5):
    (a, c) = (min(a, c), max(a, c))
    tocke = np.array([a,c])

    h = c - a # zacetna sirina intervala
    if h <= tol:
        return (a, c), tocke

    # ocena potrebnih korakov za doseg natancnosti - interval se vsakic zmanjsa za faktor 1/phi=invphi
    N = int(math.ceil(math.log(tol / h) / math.log(invphi)))

    b = a + invphi * h
    d = a + invphi2 * h
    yb = f(b)
    yd = f(d)

    n = 0
    while(h > tol or n <= N):
      n += 1
      h = invphi * h
      if yd < yb:
          c = b
          b = d
          yb = yd
          d = a + invphi2 * h
          yd = f(d)
          tockez = np.append(tocke, d)
      else:
          a = d
          d = b
          yd = yb
          b = a + invphi * h
          yb = f(b)
          tockez = np.append(tocke, d)
    if yd < yb:
        return (a, d), tockez, n
    else:
        return (d, c), tockez, n

def get_parab_min(f, a, b, c):
    if ((b - a) * (f(b) - f(c)) - (b - c) * (f(b) - f(a))) == 0:
        return None
    d = b - 0.5 * ((b - a)**2 * (f(b) - f(c)) - (b - c)**2 * (f(b) - f(a))) / ((b - a) * (f(b) - f(c)) - (b - c) * (f(b) - f(a)))
    return d

def parabmin(f, a, c, tol):

    (a, c) = (min(a, c), max(a, c))
    tockep = np.array([a,c])

    h = c - a
    if h <= tol:
        return (a, c), tockep

    b = a + invphi * h
    tockep = np.append(tockep, b)
    ya = f(a)
    yc = f(c)
    yb = f(b)

    assert ya > yb and yc > yb, f"Invalid initial interval ({a}, {c})"
    d = get_parab_min(f, a, b, c)
    yd = f(d)

    # Lahko se zgodi, da bo nova točka na intervalu [B, C]...
    if d >= b:
        d, b = b, d
        yd, yb = yb, yd

    n = 0
    while abs(c - a) > tol:
      if yd < yb:
          c = b
          b = d
          yc = yb
          yb = yd
          d = get_parab_min(f, a, b, c)
          if not d:
              break
          yd = f(d)

          if d >= b:
              d, b = b, d
              yd, yb = yb, yd
          tocke = np.append(tockep, d)
      else:
          a = d
          d = b
          ya = yd
          yd = yb
          b = get_parab_min(f, a, d, c)
          if not b:
              break
          yb = f(b)

          if d >= b:
              d, b = b, d
              yd, yb = yb, yd
          tockep = np.append(tockep, d)
      n += 1
    if yd < yb:
        return (a, d), tockep, n
    else:
        return (d, c), tockep, n

if __name__ == '__main__':

  import math

  def func(x, n):
      return x**n*np.sin(x)
  def odvod(x, n):
      return n*x**(n-1)*np.sin(x) + x**n*np.cos(x) 

  n = int(input("Vnesi red n: "))

  def novfunc(x):
    return func(x, n)
  def novodvod(x):
    return odvod(x, n)
    


  meja_a = 1. #spodnja meja
  meja_b = 10. #zgornja meja

  # "Točna" vrednost
  minimum, result = scipy.optimize.brentq(novodvod, a=3, b=7, full_output=True)
  print(f"Found minimum at {minimum}")
  print(f"\n\n{result}\n\n")

  # Uporaba vgrajene funkcije
  xmin, fval, niter, nfunc = scipy.optimize.brent(novfunc, brack=(3, 5, 7), tol=1e-15, full_output=True)
  print("Brent\n----------")
  print(f"  Mininum: {xmin}")
  print(f"  Relative Error: {abs(xmin - minimum) / minimum:.3e}")
  print(f"  Vrednost funkcije: f({xmin}) = {fval}")
  print(f"  Št poskusov: N = {niter}\n\n")

  # Zlati rez
  rezultat, tockez, koraki = zlatirez(novfunc, 2, 7, tol=1e-15)
  mean = sum(rezultat) / len(rezultat)
  print("Zlati rez\n----------")
  print(f"  Mininum: {mean}")
  print(f"  Relative Error: {abs(mean - minimum) / minimum:.3e}")
  print(f"  Vrednost funkcije: f({mean}) = {novfunc(mean)}")
  print(f"  Št poskusov: N = {koraki}\n\n")

    # Parabolicna metoda
  rezultat, tockep, koraki = parabmin(novfunc, 2, 7, tol=1e-15)
  mean = sum(rezultat) / len(rezultat)
  print("Parabolična metoda\n----------")
  print(f"  Mininum: {mean}")
  print(f"  Relative Error: {abs(mean - minimum) / minimum:.3e}")
  print(f"  Vrednost funkcije: f({mean}) = {novfunc(mean)}")
  print(f"  Št poskusov: N = {koraki}\n\n")

  plrange = np.linspace(meja_a, meja_b, 100)
  plt.plot(plrange, novfunc(plrange), "b")
  plt.grid()
  plt.plot(tockez, novfunc(tockez), "ro")
  plt.legend(("funkcija", "priblizki"), loc="lower right")
  plt.xlabel("x")
  plt.ylabel("f(x)")
  plt.title("Zlatirez")
  plt.show()

  plt.grid()
  plt.plot(plrange, novodvod(plrange), "b")
  plt.plot(tockez, novodvod(tockez), "ro")
  plt.plot(plrange, np.zeros(len(plrange)), "g-")
  plt.legend(("odvod", "priblizki"), loc="lower right")
  plt.xlabel("x")
  plt.ylabel("f'(x)")
  plt.title("Zlatirez")
  plt.show()



  plrange = np.linspace(meja_a, meja_b, 100)
  plt.plot(plrange, novfunc(plrange), "b")
  plt.grid()
  plt.plot(tockep, novfunc(tockep), "ro")
  plt.legend(("funkcija", "priblizki"), loc="lower right")
  plt.xlabel("x")
  plt.ylabel("f(x)")
  plt.title("Parabolična")
  plt.show()

  plt.grid()
  plt.plot(plrange, novodvod(plrange), "b")
  plt.plot(tockep, novodvod(tockep), "ro")
  plt.plot(plrange, np.zeros(len(plrange)), "g-")
  plt.legend(("odvod", "priblizki"), loc="lower right")
  plt.xlabel("x")
  plt.ylabel("f'(x)")
  plt.title("Parabolična")
  plt.show()
