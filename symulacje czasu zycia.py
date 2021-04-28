import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal
from sklearn.metrics import mean_squared_error 
def format_e(n):
    a = '%E' % n
    return a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]
from scipy.optimize import curve_fit

x=0.42
n0=5E14

Et=0.27
Nc=1E16
Nv=1E16
pi=1E16
tau_srh=1e-6
n=1e16    
m0=9.1093818872E-31
T=[100,120,140,160,180,200,220,240,260,280,300]
tau_rel=np.array([35,41,45,50,48,47,40,46,43,31,24,23])
rel_die=(15.15+1.65*x)
KT=[T*(1.380658E-23)/(1.60217733E-19) for T in T]
Eg=[0.417-1.28E-4*T-2.6E-7*T*T-x*(0.7+0.182+T*T*1E-9)+0.7*x*x for T in T]
mc=0.038*x*x-0.05*x+0.024
mhh=0.025
ni=[(1.35+6.5*x-1.53*0.001*T*x-6.73*x*x)*
    (1E14*np.power(Eg,0.75)*np.power(T,1.5)*
     np.exp(-1*Eg/(2*KT))) for T,Eg,KT in zip(T,Eg,KT)]
B_rad=[5.8E-13*np.power(rel_die,0.5)*np.power(1/(mc+mhh),1.5)*(1+1/mc+1/mhh)*np.power(300/T,1.5)*Eg*Eg  for T,Eg in zip(T,Eg)]

m_rat=mc/mhh
es=20.5-15.6*x+5.7*np.power(x,2)
FF=0.3
p=[ni*ni/n for ni in ni]

def SRH():
    tau=[tau_srh*(n+2*ni)/ni for ni in ni]
    return tau

SRH()

def Rad():
    tau=[1/(B_rad*(n+p)) for B_rad,p in zip(B_rad,p)]
    return tau

Rad()

def Auger():
    
    Aug_i=[(38e-18*es*es*np.power(1+m_rat,0.5)*(1+2*m_rat)*
            np.exp(((1+2*m_rat)*Eg)/((1+m_rat)*KT))
            )/(mc*FF*FF*np.power(KT/Eg,1.5)) for KT,Eg,T in zip(KT,Eg,T)]
    tau=[(2*Aug_i*(ni*ni))/(n*(n+p)) for Aug_i,ni,p in zip(Aug_i,ni,p)]
    return tau

Auger()


Et=0.27

N_=[1e12,2e12,5e12,1e13,2e13,5e13]
pi_=[2e16,4e16,5e16,7e16,9e16]
Et_=[0.18,0.19,0.2,0.21,0.22,]
tau_=[4e-8,6e-8,8e-8,1e-7,2e-7,3e-7,4e-7,]
t=np.empty(0)
mse=1e17
data=np.empty(0)
for N in N_:
    for pi in pi_:
        for tau in tau_:
            for Et in Et_:
               
                tau_rad,tau_srh,tau_tot=generate(Et,N,pi,tau)
                if mean_squared_error(tau_rel,tau_tot)<mse:
                    mse=mean_squared_error(tau_rel,tau_tot)
                    data=np.append(data,[mse,Et,N,pi,tau])
print(np.min(mse))
def func(T,Et,pi,N,tau_n):
    
    #Nc,Nv=N,N
    tau_p0,tau_n0=tau_n,tau_n
    KT=np.multiply(T,(1.380658E-23)/(1.60217733E-19))
    Eg=-0.302+np.multiply(1.93,x)+5.35*0.0001*np.multiply(T,(1-2*x))-0.81*x*x+0.832*x*x*x 
    mc=0.071*Eg*m0
    mhh=0.55*m0
    ni=(5.585-3.82*x+np.multiply(1.753*0.001,T)-1.364*0.001*np.multiply(x,T))*(1E14*np.power(Eg,0.75)*np.power(T,1.5)*
         np.exp(-1*Eg/(2*KT)))
    p0=ni/n0
    #rad
    const=5.8E-13*np.power(rel_die,0.5)*np.power(m0/(mc+mhh),1.5)*(1+m0/mc+m0/mhh)
    #n1=Nc*np.exp((Et-Eg)/KT) 
    #p1=Nv*np.exp((-Et)/KT) 
    B=np.multiply(const,np.power(ni,2))*np.power((np.multiply(300,1/T)),1.5)*(np.power(Eg,2)+3*np.multiply(KT,Eg)+3.75*np.power(KT,2)) 
    #auger
    Geei=ni*1.32e17*(mc/m0)*0.245*0.245/(np.power(rel_die,2)*np.power((1+mc/mhh),0.5)*(1+2*mc/mhh))*np.power((KT/Eg),1.5)*np.exp((-1*(1+2*mc/mhh)/(1+mc/mhh))*Eg/KT)
    Ghhi=Geei/20
    tau_rad=1e9*ni*ni/(B*(n0+pi))
    tau_srh1=1e9*(np.multiply(tau_n0,ni)+np.multiply(tau_p0,(N+ni)))/N
    #tau_srh=1e9*(tau_p0*(n0+n1+pi)+tau_n0*(p1+pi))/((n0+pi) (p0+pi))
    tau_auger=np.power(ni,3)/(n0+p0+pi)*(Geei*(n0+pi)+Ghhi)
           
    return 1/(1/tau_rad+1/tau_srh1+1/tau_auger)
T=np.array([40,50,60,70,80,90])
popt,pcov=curve_fit(func, T, tau_rel,p0=[0.18,1e13,7e12,4e-8],
            bounds=([0,5e10,1e10,1e-9],[0.4,1e16,1e14,1e-6]))
            
plt.scatter(T,tau_rel)
plt.plot(T,func(tau_rel,*popt))

tau_rad,tau_srh,tau_tot=generate(0.17,2e16,5.9e-8)


plt.scatter(T,tau_rad,color='red')
plt.scatter(T,tau_srh,color='blue')