import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as scl
import sdeint

# Set the RNG seed for reproducibility.
np.random.seed(23532532)

#
# This code is suppose to approximate
# .. math::
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


def B(z, t):
    """ Brownian motion\wiener process.

    Parameters
    ==========
    z: float
        This is the blah value.
    t: float
        The time value.

    Returns
    =======
    The blah value
    """
    if t == 0:
        return 0
    else:
        return (1/(2*np.pi*t))*(np.exp(-(z**2/2*t)))


def S(x):
    """ Solution that Dr. Yin gave me.
    """

    # FIXME: what is eps?
    x = np.array([3])
    for n in range(50):
        l = x[n] + eps[n]*x[n]*(b(alpha[n]) -
                                a(alpha[n])*x[n]) +\
            np.sqrt(eps[n]) * sigma(alpha[n]) *\
            (B(eps[n+1], n+1) - B(eps[n], n))
        w = np.append(x, l)
    return w


def g(x, t):
    """ This simulates the right part of equation (5.1).
    """
    z = x * sigma[alpha[time_to_index[t]]]
    return z


def f(x, t):
    """ Simulates left part of equation (5.1).
    """
    return x * (b[alpha[time_to_index[t]]] -
                x * a[alpha[time_to_index[t]]])


result = sdeint.stratHeun(f, g, x0, tspan)


import matplotlib.pylab as plt
plt.style.use('ggplot')

fig = plt.figure()
ax = plt.subplot()
ax.clear()
ax.plot(tspan, result)
ax.legend()
fig.tight_layout()

