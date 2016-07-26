import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as scl
import sdeint
import math as m

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


#
#This is the lognumpy.set_printoptions(threshold='nan')
#
#t = np.linspace(0, 50.0, 5000)
n = 0
log = np.linspace(0, 50.0, 5000)
while n < len(tspan):
    if tspan[n] == 0:
        log[n] = 0
        n = n+1
    else:
        log[n] = m.log(result1[n])/m.log(tspan[n])
        n = n+1
        
            
        
j = np.log(result1)
k = np.log(tspan)
h = j/k
h[0] = 0




plt.subplot(221)
plt.plot(tspan, result1, 'b')
plt.title('With Switching')

plt.subplot(222)
plt.axis([0.0,50.0,-0.4,0.4])
plt.plot(tspan, log, 'r')
plt.title('log(x(t))/log(t)')

plt.subplot(223)
plt.title('1st Diffusion')
plt.plot(tspan,result2, 'g')

plt.subplot(224)
plt.title('2nd Diffusion')
plt.plot(tspan,result3, 'k')
plt.show()

#
# This begins the code for equation 5.4. Note it uses some variables defined in the beginning like tspan for example.
#

x0 = np.matrix([[2.0], [2.0]], dtype = object)
A =[1,1]
B = [1,1]
d = 1.0
m = 1.0
eps = [1,1]
A[0] = np.matrix([[.51, 1.09], [1.2, 1.44]], dtype = object)
A[1] = np.matrix([[1.0, 0.9], [0.5, 1.5]], dtype = object)
B[0] = np.matrix([[1.5], [1.0]], dtype = object)
B[1] = np.matrix([[1.0], [1.5]], dtype = object)
eps[0] = np.matrix([[.5, 0.0], [0.0, -.5]], dtype = object)
eps[1] = np.matrix([[-0.2, 0.0], [0.0, 1.0]], dtype = object)


#
#This simulates the change in equation 5.4
#


def a1(x1,x2, t):
    """ 
        This simulates the right part of equation (5.4).
    """
    return np.matrix([[x1,0],[0,x2]]) * eps[alpha[t]]


def b1(x1,x2, t):
    """
        Simulates left part of equation (5.4).
    """
    return np.matrix([[x1,0],[0,x2]]) * (B[alpha[time_to_index[t]]] -
                                         (A[alpha[time_to_index[t]]] * np.matrix([[x1],[x2]])))



#
#
# This modules environment 1 (i.e a(1),b(1),etc) 
#
#
#
#
#

def c(x1,x2, t):
    """ 
        This simulates the right part of equation (5.5).
    """
    return np.matrix([[x1,0],[0,x2]]) * eps[0]


def d(x1,x2, t):
    """
        Simulates left part of equation (5.5).
    """
    return np.matrix([[x1,0],[0,x2]]) * (B[0] -
                                         (A[0] * np.matrix([[x1],[x2]])))



#
#
#
#
#  This begins environment 2 which is simulating 5.6 
#
#
#
#


def e(x1,x2, t):
    """ 
        This simulates the right part of equation (5.6).
    """
    return np.matrix([[x1,0],[0,x2]]) * eps[1]


def h(x1,x2, t):
    """
        Simulates left part of equation (5.6).
    """
    return np.matrix([[x1,0],[0,x2]]) * (B[1] -
                                         (A[1] * np.matrix([[x1],[x2]])))


#
#
#
#
# This will simulate 5.7
#
#
#
#

Q1 = np.asarray([[-2., 2.], [5., -5.]])


def P_trans2(t):
    res2 = scl.expm(t * Q1)
    return res2


alpha = np.empty(tspan.shape, dtype=np.int)
alpha[0] = 1
for i, t in enumerate(tspan[1:]):
    trans_prob2 = P_trans2(t)[alpha[i]]
    alpha[i+1] = np.random.binomial(1, p=trans_prob2[1])


# Create a map from continuous time to our
# discrete indices.  We'll need this to find our alpha
# in the sdeint.stratHeun function calls.
time_to_index = dict(zip(tspan, range(len(tspan))))



x01 = np.matrix([[0.1], [1.60]], dtype = object)
a =np.matrix([[2.0,4.0],[1.0,6.0]], dtype = object)
b = [1,1]
eps2 = [1,1]
b[0] = np.matrix([[4.0], [3.0]], dtype = object)
b[1] = np.matrix([[1.0], [1.0]], dtype = object)
eps2[0] = np.matrix([[0.5, 0.0], [0.0, -0.5]], dtype = object)
eps2[1] = np.matrix([[-0.2, 0.0], [0.0, 1.0]], dtype = object)


#
#This simulates the change in equation 5.7
#


def i(x1, x2, t):
    """ 
        This simulates the right part of equation (5.2).
    """
    return  eps[alpha[t]] * np.matrix([[x1,0],[0,x2]])#had to change to 2x2 matrix


def j(x1, x2, t):
    """
        Simulates left part of equation (5.2).
    """
    return np.matrix([[x1,0],[0,x2]]) * (b[alpha[time_to_index[t]]] -
                                         (a * np.matrix([[x1],[x2]]))) #had to change to 2x2 matrix







def deltaw(N, m, h):
    """Generate sequence of Wiener increments for m independent Wiener
    processes W_j(t) j=0..m-1 for each of N time intervals of length h.    

    Returns:
      dW (array of shape (N, m)): The [n, j] element has the value
      W_j((n+1)*h) - W_j(n*h) 
    """
    #l = len(N)
    W = np.random.normal(0.0, np.sqrt(h), (N,2))
    D = np.zeros((N,), dtype = object)
    for n in range(0, N):
        D[n] = np.matrix([[ W[n][0]] , [W[n][1] ]])
    return D



def stratheun(f, g, y0, tspan, dW=None):
    
#(d, m, f, G, y0, tspan, dW, __) = _check_args(f, G, y0, tspan, dW, None)
    N = len(tspan)
    h = (tspan[N-1] - tspan[0])/(N - 1)
    # allocate space for result
    y = np.zeros((N,), dtype = object)
    if dW is None:
        # pre-generate Wiener increments (for m independent Wiener processes):
        dW = deltaw(N - 1, m, h)
    y[0] = y0;
    for n in range(0, N-1):
        tn = tspan[n]
        tnp1 = tspan[n+1]
        yn = y[n]
        dWn = dW[n]#was dW[n,:]
        fn = f(yn[0,0],yn[1,0], tn) 
        Gn = g(yn[0,0],yn[1,0], tn)
        ybar = yn + h*fn + Gn*dWn#Gn.dot(dWn)
        fnbar = f(ybar[0,0],ybar[1,0], tnp1)
        Gnbar = g(ybar[0,0],ybar[1,0], tnp1)
        y[n+1] = yn + 0.5*(fn + fnbar)*h + 0.5*(Gn + Gnbar)*dWn#.dot(dWn)
    return y

result_0 = stratheun(b1, a1, x0, tspan)
result_1 = stratheun(d, c, x0, tspan)
result_2 = stratheun(h , e, x0, tspan)
result_3 = stratheun(j, i, x01, tspan)

x_1 = np.zeros(len(tspan), dtype = float)
x_2 = np.zeros(len(tspan), dtype = float)
e1_1 = np.zeros(len(tspan), dtype = float)
e1_2 = np.zeros(len(tspan), dtype = float)
e2_1 = np.zeros(len(tspan), dtype = float)
e2_2 = np.zeros(len(tspan), dtype = float)
alpha1 = np.zeros(len(tspan), dtype = float)
x3_1 = np.zeros(len(tspan), dtype = float)
x3_2 = np.zeros(len(tspan), dtype = float)
for n in range(0, len(tspan) - 1):
    x_1[n] = result_0[n][0,0]
    x_2[n] = result_0[n][1,0]
    e1_1[n] = result_1[n][0,0]
    e1_2[n] = result_1[n][1,0]
    e2_1[n] = result_2[n][0,0]
    e2_2[n] = result_2[n][1,0]
    alpha1[n] = alpha[n] + 1
    x3_1[n] = result_3[n][0,0]
    x3_2[n] = result_3[n][1,0]
    


plt.subplot(221)
plt.title('(x_1(t),x_2(t)). Phase portrait for 5.4')
plt.ylabel('x_2(t)')
plt.xlabel('x_1(t)')
plt.plot(x_1,x_2, 'k')

plt.subplot(222)
plt.axis([0.0,10.0,0.0,5])
plt.title('Component wise sample paths')
plt.ylabel('x_1(t),x_2(t)')
plt.xlabel('time')
plt.plot(tspan,x_1, 'b', label = "x_1(t)")
plt.plot(tspan,x_2, 'g', label = "x_2(t)")
plt.plot(tspan,alpha1, 'r', label = "alpha" )
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.subplot(223)
plt.axis([0.0,50.0,0.0,5])
plt.title('Environment 1')
plt.ylabel('x_1(t),x_2(t)')
plt.xlabel('time')
plt.plot(tspan,e1_1, 'b', label = "x_1(t)")
plt.plot(tspan,e1_2, 'g', label = "x_2(t)")
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.subplot(224)
plt.title('Environment 2')
plt.ylabel('x_1(t),x_2(t)')
plt.xlabel('time')
plt.axis([0.0,50.0,0.0,5])
plt.plot(tspan,e2_1, 'b')
plt.plot(tspan,e2_2, 'g')
plt.show()


plt.plot(tspan,x3_1, 'b')
plt.plot(tspan,x3_2, 'g')
plt.title('Component wise sample paths for 5.7')
plt.ylabel('x_1(t),x_2(t)')
plt.xlabel('time')
plt.show()
