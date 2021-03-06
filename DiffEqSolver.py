"""Differential Equation Solver to produce test data
Uses Goodwin Model equations
WIP
Samara Fantie 2017
Resources:
    http://matplotlib.org/2.0.0/examples/animation/basic_example.html
    http://www.danham.me/r/2015/10/29/differential-eq.html
    http://matplotlib.org/users/pyplot_tutorial.html"""

# du=u(t)(-(alpha+gamma)+(rho*v(t)))
# dv=v(t)((1/sigma)-(alpha + beta) - u(t)/sigma)

#Sample variables: (.2,.4,.2,.5,.75), (.1,.13,1.5,2.8,1.5)

from scipy.integrate import odeint
from numpy import arange
from numpy import append
from numpy import array
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def Goodwin(state,t):
    u,v = state
    du = u*(-(alpha+gamma)+(rho*v))
    dv=v*((1/sigma)-(alpha + beta) - u/sigma)
    return [du,dv]

def column(matrix, i):
    return [row[i] for row in matrix]
    
def parseit(data): #return data in proper form
    us=column(data,0)
    vs=column(data,1)
    length=len(us)
    alphabeta=append(array([alpha]*length),array([beta]*length))
    gammarho=append(array([gamma]*length),array([rho]*length))
    abgr=append(alphabeta,gammarho)
    cons=append(abgr,array([sigma]*length))
    states=append(us,vs)
    new=append(states,cons)
    return new
    
variables = input("Type values for (alpha, beta, gamma, rho,sigma): \n")
alpha, beta, gamma, rho, sigma = variables

t = arange(0,10,0.1) #adjust as needed
initial=[.5,.5] #check w/ econ
state=odeint(Goodwin, initial, t)

#state is a two-dimensional array, u in the first column, v in the second
fig1 = plt.figure()

plt.plot(state[:, 0], state[:, 1], 'b-', alpha=0.2)
plt.ylabel('V')
plt.xlabel('U')
#plt.show()

def animate(i):
    plt.plot(state[0:i, 0], state[0:i, 1], 'b-')

ani = animation.FuncAnimation(fig1, animate, interval=10)
plt.show()

#print state