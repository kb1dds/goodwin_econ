from operator import truediv
import matplotlib.pyplot as plt
import numpy as np

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

#Convering Lists to .txt
#f = open('constants_annual_final.txt', 'w')
#for element in constants_baked_A:
#    f.write((str(element)[1:len(str(element))-1]).replace(' ','')+'\n')
#f.close()

#f = open('variables_annual_final.txt', 'w')
#for element in variables_baked_A:
#    f.write((str(element)[1:len(str(element))-1]).replace(' ','')+'\n')
#f.close()



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

#constants_baked_Q = [alpha, alpha_dot, beta, beta_dot, sigma, rho, gamma]
#variables_baked_Q =[u_Q,v]

parseddata_quarterly = parsedata(u_Q, v, alpha, beta, sigma, rho, gamma)

#Convering Lists to .txt
#f = open('constants_quarterly_final.txt', 'w')
#for element in constants_baked_Q:
#    f.write((str(element)[1:len(str(element))-1]).replace(' ','')+'\n')
#f.close()

#f = open('variables_quarterly_final.txt', 'w')
#for element in variables_baked_Q:
#    f.write((str(element)[1:len(str(element))-1]).replace(' ','')+'\n')
#f.close()