import math
import matplotlib.pyplot as plt
import timeit
import numpy as np

def doubleFac(m):
    result = 1
    while m > 1:
        result *= m
        m -= 2
    return result



def erfPotencna(z):
    i = 50
    sum = 0.
    for n in range(0,i):
        summand = ((-1)**(n)*z**(2*n+1))/(math.factorial(n)*(2*n+1))
        sum += summand
    return 2/math.sqrt(math.pi) * sum


def erfAsimptotska(z):
    if z == 0:
        return None
    i = 50
    sum = 0.
    for m in range(1,i):
        summand = doubleFac(2*m-1)/((-2*z**2)**m)
        sum += summand

    return 1 - ((1 + sum)/(z*math.sqrt(math.pi)*math.exp(z**2)))

def erfRacionalna(z):
    p = 0.3275911
    a = 0.254829592
    b = -0.284496736
    c = 1.421413741
    d = -1.453152027
    e = 1.061405429
    t = 1/(1+p*z)
    y = 1.5*10**(-7)
    return 1-(a*t+b*t**2+c*t**3+d*t**4+e*t**5)*math.exp(-z**2)+y

def trueErf(z):
    return math.erf(z)







# #Grafi

z = 0.
step = 0.2
xOs = []
erfPotResults = []
erfAsiResults = []
erfRacResults = []
trueErfResults = [] 

while z <= 3.1:
    xOs.append(z)
    erfPotResults.append(erfPotencna(z))
    erfAsiResults.append(erfAsimptotska(z))
    erfRacResults.append(erfRacionalna(z))
    trueErfResults.append(trueErf(z))

    z += step

z = 3.
erfcPotResults = []
erfcAsiResults = []
erfcRacResults = []
trueErfcResults = []
step = 0.5
xOs2 =[]
while z <= 8:
    xOs2.append(z)
    erfcPotResults.append(1 - erfPotencna(z))
    erfcAsiResults.append(1 - erfAsimptotska(z))
    erfcRacResults.append(1 - erfRacionalna(z))
    trueErfcResults.append(1 - trueErf(z))
    z += step

print('_________________________________________________')
print('erf(z)=', trueErfResults)
print('__________________________________________________')
print('Potenčna vrsta: erf(z)=', erfPotResults)
print('')
print('Asimptotska vrsta: erf(z)=', erfAsiResults)
print('')
print('Racionalna aproksimacija: erf(z)=', erfRacResults)
print('_________________________________________________')
print('')
print('_________________________________________________')
print('erfc(z)=', trueErfcResults)
print('_________________________________________________')
print('Potenčna vrsta: erfc(z)=', erfcPotResults)
print('')
print('Asimptotska vrsta: erfc(z)=', erfcAsiResults)
print('')
print('Racionalna aproksimacija: erfc(z)=', erfcRacResults)
print('_________________________________________________')




plt.plot(xOs, trueErfResults)
plt.title('Python vrednosti: erf(z)')
plt.xlabel('z')
plt.ylabel('erf(z)')
plt.show()


plt.subplot(1, 4, 1)
plt.plot(xOs, trueErfResults)
plt.title('erf(z)')
plt.xlabel('z')
plt.ylabel('erf(z)')
plt.yscale("log")

plt.subplot(1, 4, 2)
plt.plot(xOs, erfPotResults)
plt.title('Potenčna vrsta: erf(z)')
plt.xlabel('z')
plt.ylabel('erf(z)')
plt.yscale("log")

plt.subplot(1, 4, 3)
plt.plot(xOs, erfAsiResults)
plt.title('Asimptotska vrsta: erf(z)')
plt.xlabel('z')
plt.ylabel('erf(z)')
plt.yscale("log")

plt.subplot(1, 4, 4)
plt.plot(xOs, erfRacResults)
plt.title('Racionalna aproksimacija: erf(z)')
plt.xlabel('z')
plt.ylabel('erf(z)')
plt.yscale("log")

plt.show()


plt.plot(xOs, trueErfResults, label = 'Phyton erf(z)', color = 'm', linewidth = 4)
plt.plot(xOs, erfPotResults, label = 'Potenčna vrsta', color = 'b') 
plt.plot(xOs, erfAsiResults, label = 'Asimptotska vrsta', color = 'g') 
plt.plot(xOs, erfRacResults, label = 'Racionalna aproksimacija', color = 'y')
plt.xlabel('z')
plt.ylabel('erf(z)')
plt.yscale("log")
plt.title('Graf erf')
plt.xlim(2, 3)
plt.ylim(0.995 , 1)
plt.legend()
plt.show()



plt.plot(xOs2, trueErfcResults)
plt.title('Python vrednosti: erfc(z)')
plt.xlabel('z')
plt.ylabel('erfc(z)')
plt.show()


plt.subplot(1, 4, 1)
plt.plot(xOs2, trueErfcResults)
plt.title('erfc(z)')
plt.xlabel('z')
plt.ylabel('erfc(z)')
plt.yscale("log")

plt.subplot(1, 4, 2)
plt.plot(xOs2, erfcPotResults)
plt.title('Potenčna vrsta: erfc(z)')
plt.xlabel('z')
plt.ylabel('erfc(z)')
plt.yscale("log")

plt.subplot(1, 4, 3)
plt.plot(xOs2, erfcAsiResults)
plt.title('Asimptotska vrsta: erfc(z)')
plt.xlabel('z')
plt.ylabel('erfc(z)')
plt.yscale("log")

plt.subplot(1, 4, 4)
plt.plot(xOs2, erfcRacResults)
plt.title('Racionalna aproksimacija: erfc(z)')
plt.xlabel('z')
plt.ylabel('erfc(z)')
plt.yscale("log")

plt.show()



plt.plot(xOs2, trueErfcResults, label = 'Phyton erfc', color ='m', linewidth = 4) 
plt.plot(xOs2, erfcPotResults, label = 'Potenčna vrsta', color = 'b') 
plt.plot(xOs2, erfcAsiResults, label = 'Asimptotska vrsta' , color = 'g')  
plt.plot(xOs2, erfcRacResults, label = 'Racionalna aproksimacija',color = 'y')
plt.xlabel('z')
plt.ylabel('erfc(z)')
plt.title('Graf erfc')
plt.yscale("log")
plt.xlim(3, 8)
plt.ylim(1e-20, 1)
plt.legend()
plt.show()








# #podprogram, sam vstaviš z
z = float(input('Vstavi število z: '))

def podobnostFunkcij(trueErf, kandidati, z):
    trueErf = trueErf(z)
    kandidati = [f(z) for f in kandidati]
    razlika = np.abs(trueErf - np.array(kandidati))
    return razlika

def podobnaF(trueErfF, kandidatiF, z):
    najbolsaF = None
    najboljsaPodobnost = float('inf')

    for func in kandidatiF:
        podobnost = podobnostFunkcij(trueErfF, [func], z)[0]
        if podobnost < najboljsaPodobnost:
            najboljsaPodobnost = podobnost
            najbolsaF = func 
    return najbolsaF

trueErfF = math.erf
kandidatiF = [erfPotencna, erfAsimptotska, erfRacionalna]

izbranaFunc = podobnaF(trueErfF, kandidatiF, z)
razlika = podobnostFunkcij(trueErfF, kandidatiF, z)
najblizjaF = izbranaFunc(z)
print('odstopanje:', razlika)
print('Najboljša metoda: ', izbranaFunc.__name__)
print('Izbrana funkcija ima vrednost: ', najblizjaF)
print('Phyton vrednost error funkcije: ', trueErfF(z))






#interval z točk, kjer se vidi katera funkcija je najbolje prikladna phyton error funkciji
zInterval = np.linspace(3.5, 4.5, 100)
najboljsaFunkcija = []
najRazlika = float('inf')


for i in range(0, len(zInterval) - 1):
    zacetekIntervala = zInterval[i]    
    najRazlika = float('inf')
    najFunc = None

    trueErf = math.erf(zacetekIntervala)
    razlika = [abs(trueErf - func(zacetekIntervala)) for func in [erfPotencna, erfAsimptotska, erfRacionalna]]
    minRazlika = min(razlika)
    najFunIndex = razlika.index(minRazlika)
    najFun = [erfPotencna, erfAsimptotska, erfRacionalna][najFunIndex]

    if minRazlika < najRazlika:
        najRazlika = minRazlika
        najFunc = najFun

    najboljsaFunkcija.append((zacetekIntervala, najFunc.__name__, najRazlika, najFun(zacetekIntervala)))
    

for interval in najboljsaFunkcija:
    print('Najbolša funkcija:', interval[1])
    print('Razlika:', interval[2])
    print('z:', interval[0])
    print('Vrednost funkcije erf pri z:', interval[3])
    print('__________________________________________________')





# #Računanje časa Racionalne aproksimacija
loop = 0
start = timeit.default_timer()
while loop < 100000:
    erfRacionalna(0)
    loop += 1

stop = timeit.default_timer()
potrebenCas0 = stop - start
print('Čas izračuna Racionalne aproksimacije: ', potrebenCas0, 's')
print('________________________________________________________________')


# #Računanje časa Asimptotske 
loop = 0
start = timeit.default_timer()
while loop < 100000:
    erfAsimptotska(0)
    loop += 1


stop = timeit.default_timer()
potrebenCas1 = stop - start
print('Čas izračuna Asimptotske vrste: ', potrebenCas1, 's')
print('________________________________________________________________')


# #Računanje časa Potenčne vrste
loop = 0
start = timeit.default_timer()
while loop < 100000:
    erfPotencna(0)
    loop += 1


stop = timeit.default_timer()
potrebenCas2 = stop - start
print('Čas izračuna Potenčne vrste: ', potrebenCas2, 's')
print('________________________________________________________________')