from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import pysheaf as py

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
    states=np.append(us,vs)
    new=[states]+[np.array([alpha]*length)]+[np.array([beta]*length)]+[np.array([gamma]*length)]+[np.array([rho]*length)]+[np.array([sigma]*length)]
    return new
    
variables = input("Type values for (alpha, beta, gamma, rho,sigma): \n")
alpha, beta, gamma, rho, sigma = variables
#example:(.1,.13,1.5,2.8,.5)

t = np.arange(0,10,0.1) #adjust as needed
initial=[.5,.5] #check w/ econ
state=odeint(Goodwin, initial, t)

#state is a two-dimensional array, u in the first column, v in the second
#uncomment to have graph plot
"""fig1 = plt.figure()
plt.plot(state[:, 0], state[:, 1], 'b-', alpha=0.2)
plt.ylabel('V')
plt.xlabel('U')
plt.show()
"""
tsmade=parseit(state)

"""CONSTANTS CODE----------------------------------------------"""

file =  open("constants_raw.txt", "r")
x = file.readlines()
x = [i[0:len(i)-1].split(',') for i in x[0:len(x)-1]]
raw_data = []
for sublist in x:
    raw_data.append([float(element) for element in sublist])
#Returns a list of lists called raw_data

hours_worked = raw_data[0]
#Hours worked by full-time and part-time employees(B4701C0A222NBEA) 
#Yearly Data: 1949-2014

Labor_Sup = raw_data[1]
Labor_Sup = [x*1000 for x in Labor_Sup]
#Now in units of people
#Civilian Employment Level (CE16OV)
#Quarterky Data: 1949-2014 

PDI = raw_data[2]
#In Billions of 2009 USD
#Real Gross Private Domestic Investment, Billions of Chained 2009 Dollars,(GPDIC1)
#Quarterly Data: 1949-2014

GDP = raw_data[3]
#In billions of 2009 USD
#Real Gross Domestic Product, Billions of Chained 2009 Dollars(GDPC1)
#Quarterly Data: 1949-2014

inf = raw_data[4]
#Consumer Price Index for All Urban Consumers: All Items (CPIAUCSL_PC1)
#(1949.01.01 - 2015.10.01) Quarterly
#Percent Change from Year Ago

unrate = raw_data[5]
#Unemployment Rate (UNRATE)
#(1949.01.01 - 2014.10.01) Quarterly

nrou = raw_data[6]
#Natural Rate of Unemployment (NROU)
#(1949.01.01 - 2014.10.01) Quarterly

u = raw_data[7]
#Worker's Share of National Income
#Shares of gross domestic income: Compensation of employees, paid: Wage and salary accruals: Disbursements (W270RE1A156NBEA)
#(1949-2014) Annual


def deriv(L):
    newL = []
    newL.append(L[1]-L[0])
    for i in range (len(L) - 2):
        newL.append((L[i+2]-L[i])/2.0)
    newL.append((L[len(L)-1]-L[len(L)-2]))
    return newL
#Derivative

def listdiv(L,K):
    newL = []
    for i in range (len(L)):
        newL.append(L[i]/K[i])
    return newL
#List Division

def listsub(L,K):
    newL =[]
    for i in range(len(L)):
        newL.append(L[i]-K[i])
    return newL
#Subtraction

def pc12(L):
    newL =[]
    for i in range (len(L)-12):
        newL.append((L[i+12]-L[i])/L[i])
    return newL
#Yearly Percent Change By Month

def pc4(L):
    newL =[]
    for i in range (len(L)-4):
        newL.append((L[i+4]-L[i])/(float(L[i])))
    return newL
#Yearly Percent Change By Quarter

def listtimes(L,K):
    newL = []
    for i in range (len(L)):
        newL.append(L[i]*K[i])
    return newL
#Multiplication
    
def lstsq(X,Y):
    N = len(X)
    m = (N*sum(listtimes(X,Y))-sum(X)*sum(Y))/(N*sum(listtimes(X,X))-sum(X)*sum(X))
    b = (sum(Y)-m*sum(X))/N
    return m,-b

def Quarterly_to_Annual(L):
    newL = []
    for i in range (int(len(L)/4)):
        newL.append((L[4*i]+L[4*i+1]+L[4*i+2]+L[4*i+3])/4)
    return newL

def Annual_to_Quarterly(L):
    newL = []
    for i in range(len(L)-1):
        m = L[i+1]-L[i]
        for j in range (4):
            newL.append(L[i]+.25*m*j)
    return newL

def parsedata(a,b,c,d,e,f,g):
    uandv=np.concatenate((np.asarray(a),np.asarray(b)))
    varlist=[uandv,np.asarray(c),np.asarray(d),np.asarray(e),np.asarray(f),np.asarray(g)]
    return varlist

#-------Aunnual Data------#
Labor_Sup_A = Quarterly_to_Annual (Labor_Sup)
PDI_A= Quarterly_to_Annual (PDI)
GDP_A = Quarterly_to_Annual (GDP)


Labor_Prod = listdiv(GDP_A, hours_worked)
#Dollars produced per hour worked
#Quarterly Data: 1949-2014

alpha = listdiv(deriv(Labor_Prod),Labor_Prod)
#Logarithmetic Derivative of Labor Productivity
beta = listdiv(deriv(Labor_Sup_A),Labor_Sup_A)
#Logarithmetic Derivative of Labor Supply
sigma = listdiv(PDI_A,GDP_A)
#Capital Output Ratio
alpha_dot = deriv(alpha)
beta_dot = deriv(beta)

#Phillips Curve#
un_inf = []
for i in range (len(inf)-4):
    un_inf.append(inf[i+4]-inf[i])
#Unanticipated Inflation

cyc_unemp = listsub(unrate,nrou)
#Cyclic Unemployment Rate
cyc_emp = [100-x for x in cyc_unemp]
#Cyclic Employment Rate

pcwage_1 =[.5307198*x+.0241174 for x in inf[0:264]]
pcwage_2 =[.5307198*x+.0241174 for x in un_inf]

v = [(100-x)/100 for x in unrate]
v_A = Quarterly_to_Annual (v)
#Employment Rate

u= [x/100 for x in u]

##1949-1966##
rho_1,gamma_1 = lstsq(v[0:68],pcwage_1[0:68])
##1967-1983##
rho_2,gamma_2 = lstsq(cyc_emp[68:132],pcwage_2[68:132])
##1983-2014##
rho_3,gamma_3 = lstsq(cyc_emp[132:264],pcwage_2[132:264])

rho = [rho_1]*17+[rho_2]*16+[rho_3]*32
gamma = [gamma_1]*17+[gamma_2]*16+[gamma_3]*32

#constants_baked_A = [alpha, beta, gamma, rho, sigma]
#variables_baked_A =[u,v_A]

parseddata_annual = parsedata(u, v_A, alpha, beta, sigma, rho, gamma)

#--------Quarterly Data-------#
hours_worked_Q = Annual_to_Quarterly(hours_worked)
u_Q = Annual_to_Quarterly (u)

Labor_Prod = listdiv(GDP, hours_worked_Q)
#Dollars produced per hour worked
#Quarterly Data: 1949-2014

alpha = listdiv(deriv(Labor_Prod),Labor_Prod)
#Logarithmetic Derivative of Labor Productivity
beta = listdiv(deriv(Labor_Sup),Labor_Sup)
#Logarithmetic Derivative of Labor Supply
sigma = listdiv(PDI,GDP)
#Capital Output Ratio
alpha_dot = deriv(alpha)
beta_dot = deriv(beta)

#Phillips Curve#
un_inf = []
for i in range (len(inf)-4):
    un_inf.append(inf[i+4]-inf[i])
#Unanticipated Inflation

cyc_unemp = listsub(unrate,nrou)
#Cyclic Unemployment Rate
cyc_emp = [100-x for x in cyc_unemp]
#Cyclic Employment Rate

pcwage_1 =[.5307198*x+.0241174 for x in inf[0:264]]
pcwage_2 =[.5307198*x+.0241174 for x in un_inf]

v = [(100-x)/100 for x in unrate]
#Employment Rate

##1949-1966##
rho_1,gamma_1 = lstsq(v[0:68],pcwage_1[0:68])
##1967-1983##
rho_2,gamma_2 = lstsq(cyc_emp[68:132],pcwage_2[68:132])
##1983-2014##
rho_3,gamma_3 = lstsq(cyc_emp[132:264],pcwage_2[132:264])

rho = [rho_1]*68+[rho_2]*64+[rho_3]*132
gamma = [gamma_1]*68+[gamma_2]*64+[gamma_3]*132


parseddata_quarterly = parsedata(u_Q, v, alpha, beta, sigma, rho, gamma)

"""MAKING THE SHEAF------------------------------------------"""

#timeseries is of the form [u0,u1,u2,...,un,v0,v1,v2,...,vn])

#functions: input is a timeseries, output is a modified time series
def pr1(ts):
    """The projection from (u,v) to (u)"""
    length=ts.size
    n=length/2
    zeros=np.zeros((n,n))
    proj=np.identity(n)
    return np.concatenate((proj,zeros))
        
def pr2(ts):
    """The projection from (u,v) to (v)"""
    length=ts.size
    n=length/2
    zeros=np.zeros((n,n))
    proj=np.identity(n)
    return np.concatenate((zeros,proj))

def eqn1(ts):
    """Goodwin equation #1 from (u,v) to v' 
    NEEDS SIGMA, ALPHA, BETA """
    #edited so all variables are arrays
    
    length=ts.size
    n=length/2
    newts=np.array([])
    for index in range(n):
        newts[:index]=ts[index+n]*(1/sigma[index]-(alpha[index]+beta[index])-(ts[index]/sigma[index])) 
    return newts
        
def eqn2(ts):
    """Goodwin equation #2 from (u,v) to u' 
    NEEDS ALPHA, GAMMA, RHO """
    #edited so all variables are arrays

    length=ts.size
    n=length/2
    newts=np.array([])
    for index in range(n):
        newts[:index]=ts[index]*(-(alpha[index]+gamma[index])+(rho[index]*ts[index+n])) 
    return newts
    
def ddt(ts):
    """Derivative of u = u' = d/dt(u) #check derivs!!!!!!!!!
    NEEDS H for f(x+h)-f(x)/h"""
    #set h as desired
    
    h=.1
    n=ts.size
    dermat=np.zeros((n,n))  #make sure this is proper construction
    for index in range(n): #go through row by row
        dermat[index,index]=(-1/h)
        if index!=(n-1):
            dermat[index,index+1]=(1/h)
    print "size of derivative is " +str(dermat.shape)
    return dermat 

def iden(ts):
    """The identity map from u to u"""
    n=ts.size
    print "size of the identity map is " +str(np.eye(n).shape)
    return np.eye(n)

def checkradii():
    consistency_radii=[s1.consistencyRadius(case) for case in input_data]
    return consistency_radii
    
#Sheaf Construction

#sheafcell=self,dimension,cofaces=[],compactClosure=True,stalkDim=None,metric=None)
#sheaf coface=(self,index,orientation,restriction)

timeseries=[tsmade,parseddata_quarterly,parseddata_annual]

#bring in the data, format [u and v,alpha, beta, sigma, rho, gamma]
#for series in timeseries:
#    ts,alpha,beta,sigma,rho,gamma=series

ts,alpha,beta,sigma,rho,gamma=tsmade

sdim=ts.size #number of samples for u & v; n+m. For n or m, use sdim/2
print "size of time series is " +str(ts.shape)

tsu=ts[:sdim/2]
tsv=ts[sdim/2:]

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
                                py.SheafCoface(index=2, orientation=-1, restriction=py.SetMorphism(eqn2(ts)))]), \
            py.SheafCell \
(dimension=0,stalkDim=(sdim/2),cofaces=[py.SheafCoface(index=0, orientation=1, restriction=py.LinearMorphism(iden(ts))), \
                                py.SheafCoface(index=2, orientation=1, restriction=py.LinearMorphism(ddt(ts)))]), \
            py.SheafCell \
(dimension=0,stalkDim=(sdim/2),cofaces=[py.SheafCoface(index=1, orientation=1, restriction=py.LinearMorphism(iden(ts))), \
                                py.SheafCoface(index=3, orientation=-1, restriction=py.LinearMorphism(ddt(ts)))])])


input_data=[py.Section([py.SectionCell(support=0,value=np.array(ts)), # U
                        py.SectionCell(support=1,value=np.array(ts)), #V
                        py.SectionCell(support=6,value=np.array(ts)), #U
                        py.SectionCell(support=7,value=np.array(ts))])] # V

# Exhibit the consistency radius of the partially-filled Section with the input data
consistency_radii=[s1.consistencyRadius(case) for case in input_data]
cr=checkradii()
print "The consistency_radii is " +str(cr)
fused_data=[s1.fuseAssignment(case) for case in input_data]
fused_consistency_radii=[s1.consistencyRadius(case) for case in fused_data]

#sample vars for input: (.1,.13,1.5,2.8,1.5)