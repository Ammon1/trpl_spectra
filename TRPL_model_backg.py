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

def ica_plot(X,a0,a1):
        y=X[:,0]*a0+X[:,1]*a1
        return y

def fit_noise(y,X_transformed):
    popt, pcov = curve_fit(ica_plot, X_transformed[15:],y[15:])
    return ica_plot(X_transformed, *popt)

def noise_remover(frame):
        intensities=frame.iloc[:,:].values
        transformer = FastICA(n_components=4,random_state=0)
        X_transformed = transformer.fit_transform(intensities)      
        X_transformed=X_transformed[:,[1,2]]
        #print(X_transformed.shape)           
        frame_ica=frame.apply(lambda y: fit_noise(y,X_transformed))
        return frame_ica.T,X_transformed

def time_constant2(x,x0,x1,A0,A1,tau0,tau1):
    return A0*np.exp(-(x-x0)/tau0)+A1*np.exp(-(x-x1)/tau1)

def time_constant1(x,x0,A0,tau0):
    return A0*np.exp(-(x-x0)/tau0)

path =r'C:\Users\Administrator\Desktop\Kacper\2020\6\19'
filename='\\10250_sa1_5us_trpl_'
frame=pd.read_csv(path+filename,sep=' ')
frame=frame-np.min(frame.iloc[:,:].values)
T=frame.iloc[:,0].values
#################### ICA
intensities=frame.iloc[:,1:].T.values
transformer = FastICA(n_components=4,random_state=0)
X_transformed = transformer.fit_transform(intensities)   
plt.plot(X_transformed[:,0],color='red')
plt.plot(X_transformed[:,1],color='blue')
plt.plot(X_transformed[:,2],color='green')
plt.plot(X_transformed[:,3],color='black')
plt.plot(X_transformed[:,4],color='pink')

plt.plot(frame.iloc[20,1:].values)#-6*X_transformed[:,2])
plt.plot(-6*X_transformed[:,2])



frame_ica,X_transformed=noise_remover(frame.iloc[:,1:].T)
plt.plot(frame_ica.iloc[10,:].values,color='red')
plt.plot(frame.iloc[10,1:].values,color='blue')
plt.ylim(-3,-1)
frame_corrected=frame.iloc[:,1:]
frame_denoised=frame_corrected-frame_ica
frame_denoised=frame_denoised-np.min(frame_denoised.iloc[:,:].values)

for i in range (190,250):
    plt.plot(np.log(frame_denoised.iloc[i,8:30].values))

    
####################

######################model
Y=np.resize(matrix,(matrix.shape[0]*matrix.shape[1],1))
X=np.tile(np.arange(0,matrix.shape[1]),matrix.shape[0])

plt.scatter(X,Y)

from sklearn.svm import SVR
model=SVR()
model.fit(X.reshape(-1,1),Y)

X_pred=X[:1200].reshape(-1,1)
Y_pred=model.predict(X_pred)
plt.plot(X_pred,Y_pred)
plt.plot(X_transformed[:,2])
###################
def fit_time(num,y_plot):
    x_plot=np.arange(0,y_plot.shape[0])
    #plt.plot(x_plot,y_plot)
    popt2=[0,0,0,0,0,0]
    try:
        if num==2:
            popt2,pcov2=curve_fit(time_constant2, x_plot, y_plot,
                                  p0=[0,0,0.1,0.1,2000,20000]
                    ,bounds=([0,0,0,0,0,0],[25,25,200,200,5000,2000000]))
            plt.plot(x_plot,time_constant2(x_plot,*popt2))
        elif num==1:
            popt1,pcov1=curve_fit(time_constant1, x_plot, y_plot,p0=[0,4,20000]
                    ,bounds=([0,0,0],[25,200,2000000]))
            plt.plot(x_plot,time_constant1(x_plot,*popt1))
    except ValueError as e:
        print(e)
    plt.plot(x_plot,y_plot)
    plt.yscale('log')
    data=[popt2[0],popt2[1],popt2[2],popt2[3],int(popt2[4]),int(popt2[5])]
    if num==2:
        return np.array(data)
    elif num==1:
        return np.array(popt1[2])

data=frame.T.apply(lambda y: fit_time(1,y[8:15]))
plt.plot(T[:220],data.iloc[:220])
