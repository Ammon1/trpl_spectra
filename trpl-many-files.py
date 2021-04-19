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

def ica_plot(X,a0,a1,a2,a3,a4,a5,a6,a7):
        y=(X[:,0]*a0+X[:,1]*a1+X[:,2]*a2+X[:,3]*a3+X[:,4]*a4+X[:,5]*a5
           +X[:,6]*a6+X[:,7]*a7)
        return y

def fit_noise(y,X_transformed):
    popt, pcov = curve_fit(ica_plot, X_transformed,y)
    return ica_plot(X_transformed, *popt)

def noise_remover(frame):
        intensities=frame.iloc[:,:].values
        transformer = FastICA(n_components=8,random_state=0)
        X_transformed = transformer.fit_transform(intensities)  
        data=frame.apply(lambda y : fit_noise(y,X_transformed))
        return X_transformed,data


def lin(x,A0,B):
    return A0*x+B

def fit_time(x_plot,y_plot):
    popt1=[0,0,0,0]
    #print(y_plot[-1])
    try:
            popt1,pcov1=curve_fit(lin, x_plot, y_plot)
            #plt.plot(x_plot,lin(x_plot,*popt1))
    except ValueError as e:
        print(e)
    #plt.plot(x_plot,y_plot)
    #print(-1/popt1[0])
    return -1/popt1[0]

def fast_result(frame):
    T=frame.iloc[:,0]
    frame=frame.iloc[:,1:]
    frame=frame-np.min(frame.iloc[:,:].values)
    time=np.arange(0,1000,1000/frame.shape[1])
    t1=90
    t2=200
    frame_log=np.log(frame)
    data=frame_log.T.apply(lambda y: fit_time(time[t1:t2],y[t1:t2]))
    plt.plot(T,data,color='red')
    return T,data

path =r'C:\Users\Administrator\Desktop\Kacper\2020\7\3\\'
filenames=['10250_5ac_1_2.4_10_3_0.79_113-300.csv',
          '10250_5ac_2.4_10_3_0.79_180-110.csv',
          '10250_5ac_2.4_10_3_0.79_113-300.csv',
          '10250_5ac_1_2.4_10_3_0.79_113-300.csv',
          '10250_5ac_2_2.4_10_3_0.79_113-300.csv']
df=pd.DataFrame({'P' : []})
for filename in filenames:
    frame=pd.read_csv(path+filename,header=None,sep=' ')
    df=pd.concat([df, frame])  
df=df.drop(['P'], axis=1)
df=df.T.apply(lambda y: y-np.min(y)).T
T,data=fast_result(df)
T,data=fast_result(frame1.T)
final=pd.concat([T, data], axis=1)
final.columns=['T','time']
final=final.groupby(['T']).mean()
plt.plot(final)

T=frame.iloc[:,0]
frame=frame.iloc[:,1:]
frame=frame-np.min(frame.iloc[:,:].values)
time=np.arange(0,1000,1000/frame.shape[1])

t1=100
t2=200
########################filtrowanie ica
X_transformed,frame1=noise_remover(df.T)
frame1=frame1-np.min(frame1.iloc[:,:].values)
#################
t1,t2=90,200
plt.plot(time[t1:t2],frame.iloc[0,t1:t2].values,color='black')
plt.plot(time[t1:t2],frame1.T.iloc[0,t1:t2].values,color='blue')
plt.yscale('log')

###########################




frame_log=np.log(frame)
frame1_log=np.log(frame1)
data=frame_log.T.apply(lambda y: fit_time(time[t1:t2],y[t1:t2]))
data1=frame1_log.apply(lambda y: fit_time(time[t1:t2],y[t1:t2]))

plt.plot(T,data,color='red')
plt.plot(T,data1,color='blue')
