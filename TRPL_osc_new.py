# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 13:31:56 2020

@author: Administrator
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.decomposition import FastICA
from scipy.optimize import curve_fit
from scipy import fftpack
from scipy import signal
from scipy import interpolate
from scipy.fftpack import fft
from scipy.signal import blackman

def ica_plot(X,a0,a1,a2,a3,a4,a5):
        y=(X[:,0]*a0+X[:,1]*a1+X[:,2]*a2+X[:,3]*a3+X[:,4]*a4+X[:,5]*a5)
        return y


def ica_plot_data(x,a0,a1,a2,a3,a4,a5):#A0,x0,tau,y0,y01):#do zmiany
        y=((X_transformed_corr[:,0]*a0+X_transformed_corr[:,1]*a1+
           X_transformed_corr[:,2]*a2+X_transformed_corr[:,3]*a3
           +X_transformed_corr[:,4]*a4+X_transformed_corr[:,5]*a5))
           #A0*np.exp(-(x-x0)/tau))
        return y

def ica_plot_data1(x,a0,a1,a2,a3,a4,a5,A0,tau):#A0,x0,tau,y0,y01):#do zmiany
        y=((X_transformed_corr[:,0]*a0+X_transformed_corr[:,1]*a1+
           X_transformed_corr[:,2]*a2+X_transformed_corr[:,3]*a3
           +X_transformed_corr[:,4]*a4+X_transformed_corr[:,5]*a5)+
           A0*np.exp(-(x+20)/tau))
        return y

def fit_noise(y,X_transformed):
    popt, pcov = curve_fit(ica_plot, X_transformed,y)
    return popt

def noise_remover(frame):
        intensities=frame.iloc[:,:].values
        transformer = FastICA(n_components=6,random_state=0)
        X_transformed = transformer.fit_transform(intensities)      
        data=frame.apply(lambda y : fit_noise(y,X_transformed))
        return X_transformed,data

def time_constant1(x,A0,tau0,B):
    return (A0*np.exp(-(x+20)/tau0)+B)

path =r'C:\Users\Administrator\Desktop\Kacper\2020\6\3'
filename='\\10350_4_sa8_trpl_'#0.145
frame=pd.read_csv(path+filename,sep=' ')
T=frame.iloc[:,0]
frame=frame.iloc[:,1:]
frame=frame-np.min(frame.iloc[:,:].values)
time=np.arange(0,100,100/frame.shape[1])
t1=5000
for i in [1200]:
    plt.plot(time[:t1],frame.iloc[i,:t1].values)
    plt.plot(time[:t1],time_constant1(time[:t1],0.15,100,3*0.01),color='red')
    plt.plot(time[:t1],time_constant1(time[:t1],*popt_fit),color='blue')
    plt.yscale('log')
    plt.ylim(2e-2,3e-1)
    
popt_fit, pcov = curve_fit(time_constant1,time[:t1],frame.iloc[50,:t1],
                            p0=[-50,0.15,100,0.03],
                            bounds=([-51,0.1,20,0.03],[-49,0.5,120,0.05]))

###########background
path =r'C:\Users\Administrator\Desktop\Kacper\2020\6\3'
filename='\\10350_4_sa8_BG_trpl_'#0.145
frame_bg=pd.read_csv(path+filename,sep=' ')
frame_bg=frame_bg.iloc[:,1:]

X_transformed,popt=noise_remover(frame_bg.T)


plt.yscale('log')
plt.ylim(2e-2,3e-1)

t0=2000
t1=5000
X_transformed_corr=X_transformed[t0:t1]
time=np.arange(0,100,100/frame.shape[1])

p0=popt.iloc[:,0].values
lb=np.where(p0<0,5*p0,0.5*p0)
ub=np.where(p0>0,5*p0,0.5*p0)
p01=np.append(p0,[0.001,30])
lb1=np.append(lb,[0.0001,20])
ub1=np.append(ub,[0.01,200])

########fit bg
X_transformed_corr=X_transformed
popt_fit, pcov = curve_fit(ica_plot_data1,time,frame_bg.iloc[0,:])
                            #p0=p01,bounds=(lb1,ub1))
plt.plot(time,ica_plot_data1(time,*popt_fit))                            
plt.plot(time,frame_bg.iloc[0,:].values)
plt.plot(time,frame_bg.iloc[0,:].values-ica_plot_data1(time,*popt_fit))
#################exp
X_transformed_corr=X_transformed[t0:t1]
popt_fit, pcov = curve_fit(ica_plot_data1,time[t0:t1],frame.iloc[1200,t0:t1],
                            p0=p01,bounds=(lb1,ub1))#
plt.plot(time[t0:t1],frame.iloc[1200,t0:t1].values)
plt.plot(time[t0:t1],ica_plot_data1(time[t0:t1],*popt_fit))
plt.plot(time[t0:t1],time_constant1(time[t0:t1],-20,popt_fit[6],popt_fit[7],0))
############

popt_fit, pcov = curve_fit(ica_plot_data1,time[t0:t1],frame.iloc[0,t0:t1],
                            p0=p01,bounds=(lb1,ub1))

plt.plot(time[t0:t1],ica_plot_data1(time[t0:t1],*popt_fit))
plt.plot(time[t0:t1],frame.iloc[0,t0:t1].values)
y_res=frame.iloc[0,t0:t1].values-ica_plot_data1(time[t0:t1],*popt_fit)
plt.plot(time[t0:t1],y_res-np.min(y_res))
plt.yscale('log')
                    

#a0,a1,a2,a3,
#A0,x0,tau,y0,y01
#    bounds=([-3e-2,2.24e-4,-5.5e-5,1e-6],
#             #1e-9,9e-1,1,-1,-1],
#            [3e-2,2.25e-4,-4e-5,3e-6])#,
#            # 20e-9,9.8e-1,300,1,0.5])
for i in [100,300,500,1000,1300]:
    time1=corrections(i)
    print(i,' ',T[i],' ',-1/time1[0])

data_new=np.empty(0)
for i in range(0,1900,1):
    data1=corrections(bounds,i)  
    data_new=np.append(data_new,[-1/data1[0],T[i]])
    print(T[i],' ',-1/data1[0])
    
def li_fit(x,A,B):
    return A*(x+30)+B

from numpy import inf

t1=100
t2=8000
t3=4000
X_transformed_corr=X_transformed[t1:t3,:]

def corrections(i):
    
    p0=popt_fit
    lb=np.where(p0<0,1.1*p0,0.9*p0)
    ub=np.where(p0>0,1.1*p0,0.9*p0)
        
    
    popt, pcov = curve_fit(ica_plot_data,time[t1:t3],
                           frame.iloc[i,t1:t3].values,p0=p0,
                           bounds=(lb,ub))

    y_real=frame.iloc[i,t1:t3].values-ica_plot_data(time[t1:t3],*popt)
    
    y_rel_log=np.log(y_real-np.min(y_real))
    y_rel_log[y_rel_log == inf]=0
    y_rel_log[y_rel_log == -inf] = 0
    popt_lin, pcov = curve_fit(li_fit,time[t1:t3],y_rel_log)
    plt.plot(time[t1:t3],y_rel_log)
    plt.plot(time[t1:t3],li_fit(time[t1:t3],*popt_lin))
    return popt_lin

data_res=data.reshape(int(data.shape[0]/2),2)
data_res_new=data_new.reshape(int(data_new.shape[0]/2),2)
plt.plot(data_res[:,1],data_res[:,0])
plt.plot(data_res_new[:,1],data_res[:,0])
plt.ylim(5,50)