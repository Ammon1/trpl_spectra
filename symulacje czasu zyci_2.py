import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error 
from scipy.optimize import curve_fit

x=0.42
n0=5E14

Et=0.27
Nc=1E16
Nv=1E16
pi=1E16
tau_srh=1e-6#parametr do symulacji
n=1e16#parametr do symulacji
m0=9.1093818872E-31
T=100
rel_die=(15.15+1.65*x)


mc=0.027#0.038*x*x-0.05*x+0.024
mhh=0.405

m_rat=mc/mhh
es=20.5-15.6*x+5.7*np.power(x,2)
FF=0.3


def SRH(tau_srh,ni):
    tau=tau_srh*(n+2*ni)/ni 
    return tau

def Rad(B_rad,n,p):
    tau=1/(B_rad*(n+p))
    return tau


def Auger(Eg,KT,ni,n,p):
    
    Aug_i=(38e-18*es*es*np.power(1+m_rat,0.5)*(1+2*m_rat)*
            np.exp(((1+2*m_rat)*Eg)/((1+m_rat)*KT))
            )/(mc*FF*FF*np.power(KT/Eg,1.5))
    tau=(2*Aug_i*(ni*ni))/(n*(n+p))
    return tau


def func(T,n,tau_srh):
    Eg=0.417-1.28E-4*T-2.6E-7*T*T-x*(0.7+0.182+T*T*1E-9)+0.7*x*x
    KT=T*(1.380658E-23)/(1.60217733E-19)
    ni=(1.35+6.5*x-1.53*0.001*T*x-6.73*x*x)*(1E14*np.power(Eg,0.75)*np.power(T,1.5)*np.exp(-1*Eg/(2*KT))) 
    B_rad=5.8E-13*np.power(rel_die,0.5)*np.power(1/(mc+mhh),1.5)*(1+1/mc+1/mhh)*np.power(300/T,1.5)*Eg*Eg
    p=ni*ni/n
    
    srh=SRH(ni)
    auger=Auger(Eg,KT,ni,n,p)
    rad=Rad(B_rad,n,p)
    
    tau_tot=1/(1/srh+1/rad+1/auger) 
    tau_tot_bez_srh=1/(1/rad+1/auger) 
    
    return tau_tot,tau_tot_bez_srh


def ni_Eg(T,n):
    Eg=0.417-1.28E-4*T-2.6E-7*T*T-x*(0.7+0.182+T*T*1E-9)+0.7*x*x
    KT=T*(1.380658E-23)/(1.60217733E-19)
    ni=(1.35+6.5*x-1.53*0.001*T*x-6.73*x*x)*(1E14*np.power(Eg,0.75)*np.power(T,1.5)*np.exp(-1*Eg/(2*KT))) 
    return Eg,ni

def x_generator(T,x,n):
    tau_srh=5e-9
    Eg=0.417-1.28E-4*T-2.6E-7*T*T-x*(0.7+0.182+T*T*1E-9)+0.7*x*x
    KT=T*(1.380658E-23)/(1.60217733E-19)
    ni=(1.35+6.5*x-1.53*0.001*T*x-6.73*x*x)*(1E14*np.power(Eg,0.75)*np.power(T,1.5)*np.exp(-1*Eg/(2*KT))) 
    B_rad=5.8E-13*np.power(rel_die,0.5)*np.power(1/(mc+mhh),1.5)*(1+1/mc+1/mhh)*np.power(300/T,1.5)*Eg*Eg
    p=ni*ni/n
    
    srh=SRH(tau_srh,ni)
    auger=Auger(Eg,KT,ni,n,p)
    rad=Rad(B_rad,n,p)
    
    tau_tot=1/(1/srh+1/rad+1/auger)
    return tau_tot

#10140
df_10140_new=df_10140.rolling(10).mean()
T,time=df_10140_new.iloc[10:,1],df_10140_new.iloc[10:,0]
popt, pcov = curve_fit(x_generator, T, time*1e-9,p0=[0.4,2e16],bounds=([0,1e16],[1,1e17]))
#dopadsowanie działa bardzo słabo-tzreba lecieć po parametrach i sprawdzać mse
plt.plot(T,time)
plt.plot(T,x_generator(T,*popt)*1e9)
print(popt)

for index,row in df_10140_new.iterrows():
        T,time=row[1],row[0]
        if T>0 and time>0 :
            popt, pcov = curve_fit(n_generator, T, time*1e-9,p0=popt)
            time_gen=n_generator(T,*popt)*1e9
            Eg,ni=ni_Eg(T,n)
            #plt.scatter(time,time_gen)
            #plt.scatter(T,time)
            plt.plot(T,time_gen)
            df_10140_new.loc[index,'n_generated']=popt
            df_10140_new.loc[index,'time_new']=time_gen
            df_10140_new.loc[index,'Eg']=Eg
            df_10140_new.loc[index,'ni']=ni
            print(0,' ',T,index,popt,time,time_gen)

plt.plot(df_10140_new.loc[:,'T'],df_10140_new.loc[:,'time_new'])
plt.plot(df_10140_new.loc[:,'T'],df_10140_new.loc[:,'time'])
plt.ylabel('time [ns]')
plt.xlabel('Temperature [K]')

plt.plot(df_10140_new.loc[:,'ni'],df_10140_new.loc[:,'n_generated']/df_10140_new.loc[:,'ni'])#correct?

plt.plot(df_10140_new.loc[:,'ni'],df_10140_new.loc[:,'n_generated'])
plt.plot(df_10140_new.loc[:,'T'],df_10140_new.loc[:,'n_generated'])
plt.plot(df_10140_new.loc[:,'T'],df_10140_new.loc[:,'ni'])
plt.ylabel('n')
plt.xlabel('ni')
plt.xscale('log')

plt.plot(1000/df_10140_new.loc[:,'T'],df_10140_new.loc[:,'n_generated'])
plt.plot(df_10140_new1.loc[:,'T'],df_10140_new1.loc[:,'n_generated'])
plt.ylim(1e16,1e17)
plt.yscale('log')       
plt.ylabel('concentration [cm-3]')
plt.xlabel('1000/Temperature [1/K]')

#10806
df_10140_new=df_10806.rolling(10).mean()
popt=2.05e17
tau_srh=1e-7#parametr do symulacji
for index,row in df_10140_new.iterrows():
        T,time=row[1],row[0]
        if 0<T<230 and time>0 : # inny parametr??
            popt, pcov = curve_fit(n_generator, T, time*1e-9,p0=popt)
            time_gen=n_generator(T,*popt)*1e9
            Eg,ni=ni_Eg(T,n)
            df_10140_new.loc[index,'n_generated']=popt
            df_10140_new.loc[index,'time_new']=time_gen
            df_10140_new.loc[index,'Eg']=Eg
            df_10140_new.loc[index,'ni']=ni
            print(0,' ',T,index,popt,time,time_gen)
        elif T>230 and time>0 :
            popt=2.05e18
            popt, pcov = curve_fit(n_generator, T, time*1e-9,p0=popt)
            time_gen=n_generator(T,*popt)*1e9
            Eg,ni=ni_Eg(T,n)
            df_10140_new.loc[index,'n_generated']=popt
            df_10140_new.loc[index,'time_new']=time_gen
            df_10140_new.loc[index,'Eg']=Eg
            df_10140_new.loc[index,'ni']=ni
            print(0,' ',T,index,popt,time,time_gen)

plt.plot(df_10140_new.loc[:,'T'],df_10140_new.loc[:,'time_new'])
plt.plot(df_10140_new.loc[:,'T'],df_10140_new.loc[:,'time'])
plt.ylabel('time [ns]')
plt.xlabel('Temperature [K]')

plt.plot(df_10140_new.loc[:,'ni'],df_10140_new.loc[:,'n_generated']/df_10140_new.loc[:,'ni'])#correct?

plt.plot(df_10140_new.loc[:,'ni'],df_10140_new.loc[:,'n_generated'])
plt.ylabel('n/ni')
plt.xlabel('ni')
plt.xscale('log')

plt.plot(1000/df_10140_new.loc[:,'T'],df_10140_new.loc[:,'n_generated'])
plt.plot(df_10140_new.loc[:,'T'],df_10140_new.loc[:,'n_generated'])
plt.ylim(1e16,1e17)
plt.yscale('log')       
plt.ylabel('concentration [cm-3]')
plt.xlabel('1000/Temperature [1/K]')

#10141
df_10140_new=df_10141_1.rolling(10).mean()
popt=0.4
tau_srh=1e-7#parametr do symulacji

popt, pcov = curve_fit(x_generator, T, time*1e-9,p0=popt)

for index,row in df_10140_new.iterrows():
        T,time=row[1],row[0]
        if 0<T<230 and time>0 : # inny parametr??
            popt, pcov = curve_fit(x_generator, T, time*1e-9,p0=popt)
            time_gen=x_generator(T,*popt)*1e9
            Eg,ni=ni_Eg(T,n)
            df_10140_new.loc[index,'x_generated']=popt
            df_10140_new.loc[index,'time_new']=time_gen
            df_10140_new.loc[index,'Eg']=Eg
            print(0,' ',T,index,popt,time,time_gen)
        elif T>230 and time>0 :
            popt=2.05e18
            popt, pcov = curve_fit(n_generator, T, time*1e-9,p0=popt)
            time_gen=n_generator(T,*popt)*1e9
            Eg,ni=ni_Eg(T,n)
            df_10140_new.loc[index,'n_generated']=popt
            df_10140_new.loc[index,'time_new']=time_gen
            df_10140_new.loc[index,'Eg']=Eg
            df_10140_new.loc[index,'ni']=ni
            print(0,' ',T,index,popt,time,time_gen)

plt.plot(df_10140_new.loc[:,'T'],df_10140_new.loc[:,'time_new'])
plt.plot(df_10140_new.loc[:,'T'],df_10140_new.loc[:,'time'])
plt.ylabel('time [ns]')
plt.xlabel('Temperature [K]')

plt.plot(df_10140_new.loc[:,'ni'],df_10140_new.loc[:,'n_generated']/df_10140_new.loc[:,'ni'])#correct?

plt.plot(df_10140_new.loc[:,'ni'],df_10140_new.loc[:,'n_generated'])
plt.ylabel('n/ni')
plt.xlabel('ni')
plt.xscale('log')

plt.plot(1000/df_10140_new.loc[:,'T'],df_10140_new.loc[:,'n_generated'])
plt.plot(df_10140_new.loc[:,'T'],df_10140_new.loc[:,'n_generated'])
plt.ylim(1e16,1e17)
plt.yscale('log')       
plt.ylabel('concentration [cm-3]')
plt.xlabel('1000/Temperature [1/K]')

#10251
df_10140_new=df_10251.rolling(10).mean()
popt=2.05e18
tau_srh=1e-7#parametr do symulacji
for index,row in df_10140_new.iterrows():
        T,time=row[1],row[0]
        if 0<T<250 and time>0 : # inny parametr??
            popt, pcov = curve_fit(n_generator, T, time*1e-9,p0=popt)
            time_gen=n_generator(T,*popt)*1e9
            Eg,ni=ni_Eg(T,n)
            df_10140_new.loc[index,'n_generated']=popt
            df_10140_new.loc[index,'time_new']=time_gen
            df_10140_new.loc[index,'Eg']=Eg
            df_10140_new.loc[index,'ni']=ni
            print(0,' ',T,index,popt,time,time_gen)
        elif T>250 and time>0 :
            popt=2.05e18
            popt, pcov = curve_fit(n_generator, T, time*1e-9,p0=popt)
            time_gen=n_generator(T,*popt)*1e9
            Eg,ni=ni_Eg(T,n)
            df_10140_new.loc[index,'n_generated']=popt
            df_10140_new.loc[index,'time_new']=time_gen
            df_10140_new.loc[index,'Eg']=Eg
            df_10140_new.loc[index,'ni']=ni
            print(0,' ',T,index,popt,time,time_gen)

plt.plot(df_10140_new.loc[:,'T'],df_10140_new.loc[:,'time_new'])

def f(x):    
 return 1e9*n_generator(np.abs(x[1]),np.abs(x[2]))  

plt.plot(df_10140_new.loc[:,'T'],df_10140_new.loc[:,'time'])
df_10140_new['time_new1']=df_10140_new.apply(f,axis=1)
plt.ylabel('time [ns]')
plt.xlabel('Temperature [K]')

plt.plot(df_10140_new.loc[:,'ni'],df_10140_new.loc[:,'n_generated']/df_10140_new.loc[:,'ni'])#correct?

plt.plot(df_10140_new.loc[:,'ni'],df_10140_new.loc[:,'n_generated'])
plt.ylabel('n/ni')
plt.xlabel('ni')
plt.xscale('log')

plt.plot(1000/df_10140_new.loc[:,'T'],df_10140_new.loc[:,'n_generated'])
plt.plot(df_10140_new.loc[:,'T'],df_10140_new.loc[:,'n_generated'])
plt.ylim(1e16,1e17)
plt.yscale('log')       
plt.ylabel('concentration [cm-3]')
plt.xlabel('1000/Temperature [1/K]')

#insb ni simmulation
def f_ni(T):
    Eg=0.417-1.28E-4*T-2.6E-7*T*T-x*(0.7+0.182+T*T*1E-9)+0.7*x*x
    KT=T*(1.380658E-23)/(1.60217733E-19)
    ni1=5.76E14*np.power(T,1.5)*np.exp(-0.129/KT)
    ni2=(1.35+6.5*x-1.53*0.001*T*x-6.73*x*x)*(1E14*np.power(Eg,0.75)*np.power(T,1.5)*np.exp(-1*Eg/(2*KT))) 
    return ni1,ni2


           
 
