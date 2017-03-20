#Goodwin model sheafification
#2/7/2017 Samara Fantie & co.

#import pandas as pd
import pysheaf as py
import numpy as np

#parser for data- will add when data is found

#structure
#coface:  def __init__(self,index,orientation) 
#cell: def __init__(self,dimension,compactClosure=True,cofaces=[], name=None)

#timeseries is of the form [u0,u1,u2,...,un,v0,v1,v2,...,vn])

#functions: input is a timeseries, output is a modified time series
def pr1(ts):
    """The projection from (u,v) to (u)"""
    length=np.ts.size
    n=length/2
    zeros=np.array([0]*n)
    proj=np.identity(n)
    #for index in range(n):
    #    proj[index]=np.append(proj[index],zeros) #fix
    #return proj
    return np.concatenate((proj,zeros))
        
def pr2(ts):
    """The projection from (u,v) to (v)"""
    length=np.ts.size
    n=length/2
    zeros=np.array([0]*n)
    proj=np.identity(n)
    return np.concatenate((zeros,proj))

def eqn1(ts):
    """Goodwin equation #1 from (u,v) to v' 
    NEEDS SIGMA, ALPHA, BETA """
    length=np.ts.size
    n=length/2
    newts=np.array([])
    for index in range(n):
        newts[:,index]=ts[index+n]*(1/sigma-(alpha+beta)-(ts[index]/sigma)) 
    return newts
        
def eqn2(alpha,gamma,rho):
    """Goodwin equation #2 from (u,v) to u' 
    NEEDS ALPHA, GAMMA, RHO """
    def eqn2b(ts):
        length=np.ts.size
        n=length/2
        newts=np.array([])
        for index in range(n):
            newts[:,index]=ts[index]*(-(alpha+gamma)+(rho*ts[index+n])) 
        return newts

def ddt(ts):
    """Derivative of u = u' = d/dt(u) #check derivs!!!!!!!!!
    NEEDS H for f(x+h)-f(x)/h"""
    n=np.ts.size
    dermat=np.array([])
    for index in range(n): #go through row by row
        dermat[index,index]=(-1/h)
        if index!=n:
            dermat[index,index+1]=(1/h)
    return dermat 

def iden(ts):
    """The identity map from u to u"""
    n=np.ts.size
    return np.eye(n)

#Sheaf Construction

#sheafcell=self,dimension,cofaces=[],compactClosure=True,stalkDim=None,metric=None)
#sheaf coface=(self,index,orientation,restriction)

sdim=np.ts.size #number of samples for u & v; n+m. For n or m, use sdim/2
#didn't like having it declared here

s1=py.Sheaf([py.SheafCell(dimension=1,stalkDim=(sdim/2),cofaces=[]), \
            py.SheafCell(dimension=1,stalkDim=(sdim/2),cofaces=[]), \
            py.SheafCell(dimension=1,stalkDim=(sdim/2),cofaces=[]), \
            py.SheafCell(dimension=1,stalkDim=(sdim/2),cofaces=[]), \
            py.SheafCell \
(dimension=0,stalkDim=sdim,cofaces=[py.SheafCoface(index=0, orientation=1, restriction=py.LinearMorphism(pr1(ts))), \
                                py.SheafCoface(index=1, orientation=1, restriction=py.LinearMorphism(pr2(ts))), \
                                py.SheafCoface(index=3, orientation=1, restriction=py.SetMorphism(eqn1(ts)))]), \
            py.SheafCell \
(dimension=0,stalkDim=sdim,cofaces=[py.SheafCoface(index=0, orientation=-1, restriction=py.LinearMorphism(pr1(ts))), \
                                py.SheafCoface(index=1, orientation=-1, restriction=py.LinearMorphism(pr2(ts))), \
                                py.SheafCoface(index=2, orientation=-1, restriction=py.SetMorphism(eqn2b(ts)))]), \
            py.SheafCell \
(dimension=0,stalkDim=(sdim/2),cofaces=[py.SheafCoface(index=0, orientation=1, restriction=py.LinearMorphism(iden(ts))), \
                                py.SheafCoface(index=2, orientation=1, restriction=py.LinearMorphism(ddt(ts)))]), \
            py.SheafCell \
(dimension=0,stalkDim=(sdim/2),cofaces=[py.SheafCoface(index=1, orientation=1, restriction=py.LinearMorphism(iden(ts))), \
                                py.SheafCoface(index=3, orientation=-1, restriction=py.LinearMorphism(ddt(ts)))])])

