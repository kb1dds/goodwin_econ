from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

"""---DIFFERENTIAL EQUATION SOLVER TO MAKE DATA_______________________________________"""

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
    alphabeta=np.append(np.array([alpha]*length),np.array([beta]*length))
    gammarho=np.append(np.array([gamma]*length),np.array([rho]*length))
    abgr=np.append(alphabeta,gammarho)
    cons=np.append(abgr,np.array([sigma]*length))
    states=np.append(us,vs)
    new=np.append(states,cons)
    return new
    
variables = input("Type values for (alpha, beta, gamma, rho,sigma): \n")
alpha, beta, gamma, rho, sigma = variables
#example:(.1,.13,1.5,2.8,.5)

t = np.arange(0,15,0.1) #adjust as needed
initial=[.5,.5] #check w/ econ
state=odeint(Goodwin, initial, t)

#state is a two-dimensional array, u in the first column, v in the second
fig1 = plt.figure()
plt.plot(state[:, 0], state[:, 1], 'b-', alpha=0.2)
plt.ylabel('V')
plt.xlabel('U')
plt.show()

tsmade=parseit(state)

"""CONSTANTS CODE----------------------------------------------"""

