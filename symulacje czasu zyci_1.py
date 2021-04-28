import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal
from sklearn.metrics import mean_squared_error 
def format_e(n):
    a = '%E' % n
    return a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]
from scipy.optimize import curve_fit
import pandas as pd

x=0.42
n0=5E14

Et=0.27
Nc=1E16
Nv=1E16

tau_srh=1e-6#parametr do symulacji
#n=1e16#parametr do symulacji
m0=9.1093818872E-31
T=np.arange(20,300,1)
tau_rel=np.array([35,41,45,50,48,47,40,46,43,31,24,23])
rel_die=(15.15+1.65*x)


mhh=0.405

m_rat=mc/mhh
es=20.5-15.6*x+5.7*np.power(x,2)
FF=0.3

def n1(N,Eg,E,KT):
    return N*np.exp((E-Eg)/KT)

def SRH(ni,N,Eg,E,KT):
    tau=[tau_srh*(n+2*ni+n1(N,Eg,E,KT))/ni for (KT,ni,Eg) in zip(KT,ni,Eg)]
    return tau



def Rad(B_rad,n):
    KT,ni,Eg,p=pre(T,n)
    tau=[1/(B_rad*(n+p)) for B_rad,p in zip(B_rad,p)]
    return tau



def Auger(n):
    KT,ni,Eg,p= pre(T,n)
    Aug_i=[(38e-18*es*es*np.power(1+m_rat,0.5)*(1+2*m_rat)*
            np.exp(((1+2*m_rat)*Eg)/((1+m_rat)*KT))
            )/(mc*FF*FF*np.power(KT/Eg,1.5)) for KT,Eg,T in zip(KT,Eg,T)]
    tau=[(2*Aug_i*(ni*ni))/(n*(n+p)) for Aug_i,ni,p in zip(Aug_i,ni,p)]
    return tau

def pre(T,n):
    KT=[T*(1.380658E-23)/(1.60217733E-19) for T in T]
    Eg=[0.417-1.28E-4*T-2.6E-7*T*T-x*(0.7+0.182+T*T*1E-9)+0.7*x*x for T in T]
    ni=[(1.35+6.5*x-1.53*0.001*T*x-6.73*x*x)*
    (1E14*np.power(Eg,0.75)*np.power(T,1.5)*
     np.exp(-1*Eg/(2*KT))) for T,Eg,KT in zip(T,Eg,KT)]
    p=[ni*ni/n for ni in ni]
    return KT,ni,Eg,p

def func(T,n,tau_srh,x,E,N,a):
    mc=0.038*x*x-0.05*x+0.024+a
    KT=[T*(1.380658E-23)/(1.60217733E-19) for T in T]
    Eg=[0.417-1.28E-4*T-2.6E-7*T*T-x*(0.7+0.182+T*T*1E-9)+0.7*x*x for T in T]
    ni=[(1.35+6.5*x-1.53*0.001*T*x-6.73*x*x)*
    (1E14*np.power(Eg,0.75)*np.power(T,1.5)*
     np.exp(-1*Eg/(2*KT))) for T,Eg,KT in zip(T,Eg,KT)]
    #n=5.81E16
    p=[ni*ni/n for ni in ni]
    
    B_rad=[5.8E-13*np.power(rel_die,0.5)*np.power(1/(mc+mhh),1.5)*(1+1/mc+1/mhh)*np.power(300/T,1.5)*Eg*Eg  for T,Eg in zip(T,Eg)]
    
    srh=SRH(ni,N,Eg,E,KT)
    auger=Auger(n)
    rad=Rad(B_rad,n)
    
    tau_tot=[1/(1/srh+1/rad+1/auger) for srh,rad,auger in zip(srh,rad,auger)]
    tau_tot_bez_srh=[1/(1/rad+1/auger) for rad,auger in zip(rad,auger)]
    
    return tau_tot,tau_tot_bez_srh,rad,srh,auger
from sklearn.metrics import mean_absolute_error

def tau(n,tau_srh,x):
    tau_tot,tau_tot_bez_srh,rad,srh,auger=func(T,n,tau_srh,x)
    tau_tot=[tau_tot*1e9 for tau_tot in tau_tot]
    return mean_absolute_error(tau_tot, time)
    
df_10140_new=df_10140.rolling(10).mean()
T,time=df_10140_new.iloc[10:,1],df_10140_new.iloc[10:,0]#T -90

nn=np.logspace(15,17,num=10)
n_srh=np.logspace(-4,-9,num=10)
xx=np.arange(0,1.1,0.05)
xx_long=np.repeat(xx,(nn.shape[0]*n_srh.shape[0]))
nn_long=np.tile(np.repeat(nn,(n_srh.shape[0])),xx.shape[0])
n_srh_long=np.tile(n_srh,int(nn.shape[0]*xx.shape[0]))

df[3]=df.apply(lambda row: tau(row[0],row[1],row[2]),axis=1)

#generacja dopasowania
df=pd.DataFrame([nn_long,n_srh_long,xx_long]).T

#generacja krzyych
T=np.arange(10,300,1)
x=0.145
n=3e16
tau_srh=1e-7
for E in [0.1,0.5,1,1.5,2,2.5]:
  for N in [100,1000]:
      for tau_srh in [1e-7,1e-6,1e-5]:
            tau_tot,tau_tot_bez_srh,rad,srh,auger=func(T,n,tau_srh,x,E,N)
            tau_tot=[tau_tot*1e9 for tau_tot in tau_tot]
            #print(tau_tot)
            plt.plot(T,tau_tot,label=[E,N])
            plt.ylim(1,100)
            #plt.xlim(150,250)
            #plt.yscale('log')
            plt.legend()
            
#generacja krzyych
T=np.arange(20,300,1)
x=0.145
n=3e16
tau_srh=1e-8
N=100
E=1
nn=np.logspace(16,17,num=10)
for x in [0.145]:
  for n in [4e16,5e16]:
      #for tau_srh in [1e-7,1e-6,1e-5]:
      if n>3.6e16:
            tau_tot,tau_tot_bez_srh,rad,srh,auger=func(T,n,tau_srh,x,E,N)
            
            tau_tot=[tau_tot*1e9 for tau_tot in tau_tot]
            rad=[rad*1e9 for rad in rad]
            auger=[auger*1e9 for auger in auger]
            #print(tau_tot)
            plt.plot(T,tau_tot,label='tot')
            plt.plot(T,rad,label='rad')
            plt.plot(T,auger,label='auger')
            plt.ylim(1,1000)
            #plt.xlim(150,250)
            plt.yscale('log')
            plt.legend()
            
for x in [0.4]:
  for n in [4e16,5e16]:
      #for tau_srh in [1e-7,1e-6,1e-5]:
      if n>3.6e16:
            tau_tot,tau_tot_bez_srh,rad,srh,auger=func(T,n,tau_srh,x,E,N)
            
            tau_tot=[tau_tot*1e9 for tau_tot in tau_tot]
            rad=[rad*1e9 for rad in rad]
            auger=[auger*1e9 for auger in auger]
            
            #print(tau_tot)
            plt.plot(T,tau_tot,label='tot')
            plt.plot(T,rad,label='rad')
            plt.plot(T,auger,label='auger')
            plt.ylim(1,1000)
            #plt.xlim(150,250)
            plt.yscale('log')
            plt.legend()
            
            

