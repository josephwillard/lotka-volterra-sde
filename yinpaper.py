import sympy as sp
import matplotlib.pyplot as plt
import numpy as np
import random as r
import scipy.linalg as sc
import sdeint
import pysde
import math as m
"""
This code is suppose to approximate dx(t) = x(t)[b(\alpha(t))-a(\alpha(t))]dt + x(t)\sigma(\alpha(t)) \circ dW(t) in Dr. Yin's paper (example 5.1). I am using the sdeint package that gives a variety of methods to integrate SDEs. 
"""


np.set_printoptions(threshold='nan')
tspan = np.linspace(0,50.0,5000) # np.arange(0,50,1)
#t = np.arange(0,50,1)
#eps = np.array(1/t2, dtype = float)
#eps[0]= 0
x=0
alpha = [r.randint(1,2) for x in xrange(5000)]
alpha[0]=2
#x0= np.zeros((50,),dtype = float)
#x0[0] = 3
x0=3


def sigma(x):
    if x == 1:
        return 0.2
    else:
        return 0.0

def b(x):
    if x ==1:
        return 3.0
    else:
        return 2.0

def a(x):
    if x ==1:
        return 4.0
    else:
        return 1.0

#Brownian motion\wiener process
def B(z,t):
    if t == 0:
        return 0
    else:
        return (1/(2*np.pi*t))*(np.exp(-(z**2/2*t)))


#Solution that Dr. Yin gave me.
def S(x):
    x = [3]
    for n in range(50):
        l = x[n] + eps[n]*x[n]*(b(alpha[n])- a(alpha[n])*x[n]) + np.sqrt(eps[n])*sigma(alpha[n])*(B(eps[n+1],n+1) - B(eps[n],n))
        w = np.append(x,l)
    return w

#this simulates the right part of equation 
def g(x,t):
     z = x*sigma(alpha[int(m.ceil(t))])
     #b = np.asscalar(z)
     return z

#Simulates left part.
def f(x, t):
    l = []
    for t in range(50):
        w = b(alpha[int(m.ceil(t))])-a(alpha[int(m.ceil(t))])
        l = np.append(l,w)
    return x*(l[t]*x) #S(x)*(b(x)-a(x)*S(x)) 

    
result = sdeint.stratHeun(f, g, x0, tspan)

plt.plot(tspan, result)
plt.show()

