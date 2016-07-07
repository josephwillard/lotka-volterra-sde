import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as scl
import sdeint

# Set the RNG seed for reproducibility.
np.random.seed(23532532)

#
# This code is suppose to approximate,
#
#   dx(t) = x(t)[b(\alpha(t))-a(\alpha(t))]dt + x(t)\sigma(\alpha(t)) \circ dW(t)
#
# in Dr. Yin's paper (example 5.1). I am using the sdeint package that gives a
# variety of methods to integrate SDEs.
#


# Our discretized times
tspan = np.linspace(0, 50.0, 5000)

#
# Simulate a simple continuous, two-state time Markov
# Chain.
#
Q = np.asarray([[-2., 2.], [3., -3.]])


def P_trans(t):
    res = scl.expm(t * Q)
    return res

alpha = np.empty(tspan.shape, dtype=np.int)
alpha[0] = 1
for i, t in enumerate(tspan[1:]):
    trans_prob = P_trans(t)[alpha[i]]
    alpha[i+1] = np.random.binomial(1, p=trans_prob[1])

# Create a map from continuous time to our
# discrete indices.  We'll need this to find our alpha
# in the sdeint.stratHeun function calls.
time_to_index = dict(zip(tspan, range(len(tspan))))

x0 = 3
sigma = np.array([0.2, 0.0])
a = np.array([4.0, 1.0])
b = np.array([3.0, 2.0])

#
#This simulates the 1st diffusion.
#
def c(x, t):    
    return x * (3.0 - 4.0 * x)

def d(x, t):
    return x * 0.2

    

#
#This simulates switching.
#
def g(x, t):
    #This simulates the right part of equation (5.1).
    return  x * sigma[alpha[time_to_index[t]]]


def f(x, t):
    #Simulates left part of equation (5.1).
    return x * (b[alpha[time_to_index[t]]] - x * a[alpha[time_to_index[t]]])

#
#This simulates the 2nd diffusion.
#

def e(x, t):
    #This simulates the left part of equation (5.1).
    return  x * (2 - x)

def h(x, t):
    #Simulates right part of equation (5.1).
    return 0


result1 = sdeint.stratHeun(f, g, x0, tspan) #Switching
result2 = sdeint.stratHeun(c, d, x0, tspan) #1st diffusion
result3 = sdeint.stratHeun(e, h, x0, tspan) #2nd diffusion

log = np.log(result1)/np.log(tspan)


plt.subplot(221)
plt.plot(tspan, result1)
plt.title('With Switching')

plt.subplot(222)
plt.plot(tspan, log)
plt.title('log(x(t))/log(t)')

plt.subplot(223)
plt.title('1st Diffusion')
plt.plot(tspan,result2)

plt.subplot(224)
plt.title('2nd Diffusion')
plt.plot(tspan,result3)
plt.show()

